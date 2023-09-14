# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import defaultdict
from deepspeed.launcher.runner import fetch_hostfile
from deepspeed.accelerator import get_accelerator

import mii
from mii.utils import get_num_gpus, get_provider_name
from mii.logging import logger
from mii.config import ReplicaConfig, LoadBalancerConfig


def config_to_b64_str(config):
    # convert json str -> bytes
    json_bytes = config.json().encode()
    # base64 encoded bytes
    b64_config_bytes = base64.urlsafe_b64encode(json_bytes)
    # bytes -> str
    return b64_config_bytes.decode()


class MIIServer():
    '''Initialize the model, setup the server for the model under model_path'''
    def __init__(self,
                 deployment_name,
                 task_name,
                 model_name,
                 model_path,
                 ds_optimize=True,
                 ds_zero=False,
                 ds_config=None,
                 mii_configs={},
                 lb_config=None):
        mii_configs = mii.config.MIIConfig(**mii_configs)

        self.task = mii.utils.get_task(task_name)

        self.num_gpus = get_num_gpus(mii_configs)
        assert self.num_gpus > 0, "GPU count must be greater than 0"

        self.port_number = mii_configs.port_number

        if not os.path.isfile(mii_configs.hostfile):
            logger.info(f"Hostfile {mii_configs.hostfile} not found, creating hostfile.")
            num_gpu = get_accelerator().device_count()
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                temp_file.write(f"localhost slots={num_gpu}")
            mii_configs.hostfile = temp_file.name

        lb_config = self._create_lb_config(mii_config=mii_configs)

        processes = self._initialize_service(deployment_name,
                                             model_name,
                                             model_path,
                                             ds_optimize,
                                             ds_zero,
                                             ds_config,
                                             mii_configs,
                                             lb_config)
        self._wait_until_server_is_live(processes, lb_config.replica_configs)

    def _create_lb_config(self, mii_config):
        # add fields for replica deployment
        replica_pool = _allocate_processes(mii_config.hostfile,
                                           mii_config.tensor_parallel,
                                           mii_config.replica_num)
        if mii_config["deploy_rank"]!=-1 and mii_config.replica_num==1:
            if isinstance(mii_config["deploy_rank"], int):
                replica_pool = [(replica_pool[0][0], [mii_config["deploy_rank"]])]
            elif isinstance(mii_config["deploy_rank"], list):
                replica_pool = [(replica_pool[0][0], mii_config["deploy_rank"])]
            else:
                raise ValueError(
                    f"Expected a int or list of GPU indices to deploy on. Received: {mii_config['deploy_rank']}"
                )
        replica_configs = []
        for i, (hostname, gpu_indices) in enumerate(replica_pool):
            # Reserver port for a LB proxy when replication is enabled
            port_offset = 1
            base_port = mii_config.port_number + i * mii_config.tensor_parallel + port_offset
            tensor_parallel_ports = list(
                range(base_port,
                      base_port + mii_config.tensor_parallel))
            torch_dist_port = mii_config.torch_dist_port + i
            replica_configs.append(
                ReplicaConfig(hostname=hostname,
                              tensor_parallel_ports=tensor_parallel_ports,
                              torch_dist_port=torch_dist_port,
                              gpu_indices=gpu_indices))

        lb_config = LoadBalancerConfig(port=mii_config.port_number,
                                       replica_configs=replica_configs)

        return lb_config

    def _wait_until_server_is_live(self, processes, deployment):
        for process, repl_config in zip(processes, deployment):
            sockets_open = False
            while not sockets_open:
                sockets_open = all(
                    self._is_socket_open(repl_config.hostname,
                                         port)
                    for port in repl_config.tensor_parallel_ports)
                process_alive = self._is_server_process_alive(process)
                if not process_alive:
                    raise RuntimeError(
                        "server crashed for some reason, unable to proceed")
                time.sleep(4)
                logger.info("waiting for server to start...")
            logger.info(
                f"server has started on ports {repl_config.tensor_parallel_ports}")

    def _is_socket_open(self, host, port):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0

    def _is_server_process_alive(self, process):
        if process is None:
            return True
        try:
            process.wait(1)
        except subprocess.TimeoutExpired as err:
            # timeout means we're still running and all (probably) okay
            is_alive = True
        else:
            # no exception case
            is_alive = False
        return is_alive

    def _build_server_args(self,
                           deployment_name,
                           model_name,
                           model_path,
                           ds_optimize,
                           ds_zero,
                           ds_config,
                           mii_configs,
                           port):
        # serialize mii config
        b64_config_str = config_to_b64_str(mii_configs)

        server_args_str = f"--deployment-name {deployment_name} --task-name {mii.utils.get_task_name(self.task)} --model {model_name} --model-path {model_path} --port {port}"
        server_args_str += " --ds-optimize" if ds_optimize else ""

        # XXX: fetch model provider based on model name in a more general way
        provider = get_provider_name(model_name, self.task)
        server_args_str += f" --provider {provider}"

        server_args_str += f" --config {b64_config_str}"
        server_args_str += " --ds-zero" if ds_zero else ""
        if ds_zero and ds_config is not None:
            if isinstance(ds_config, dict):

                def create_config_from_dict(tmpdir, config_dict):
                    if not os.path.exists(tmpdir):
                        os.makedirs(tmpdir)
                    config_path = os.path.join(tmpdir, 'temp_config.json')
                    with open(config_path, 'w') as fd:
                        json.dump(config_dict, fd)
                    return config_path

                model_dir = Path(model_path).parent.resolve()
                ds_config_path = create_config_from_dict(model_dir, ds_config)
            elif isinstance(ds_config, str):
                ds_config_path = ds_config
            else:
                raise ValueError(
                    f"Expected a string path to an existing deepspeed config, or a dictionary. Received: {ds_config}"
                )
            server_args_str += f" --ds-config {ds_config_path}"
        printable_config = f"task-name {mii.utils.get_task_name(self.task)} model {model_name} model-path {model_path} port {self.port_number} provider {provider}"
        logger.info(f"MII using multi-gpu deepspeed launcher:\n" +
                    self.print_helper(printable_config))
        return server_args_str

    def print_helper(self, args):
        # convert to list
        args = args.split(" ")
        # convert to dict
        dct = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
        printable_string = ""
        printable_string += " " + "-" * 60 + "\n"
        for k, v in dct.items():
            dots = "." * (29 - len(k))
            printable_string += f" {k} {dots} {v} \n"
        printable_string += " " + "-" * 60
        return printable_string

    def _launch_load_balancer(self,
                              deployment_name,
                              model_name,
                              model_path,
                              ds_optimize,
                              ds_zero,
                              ds_config,
                              mii_configs,
                              lb_config):

        # serialize mii config
        b64_config_str = config_to_b64_str(lb_config)

        return self._launch_server_process(
            deployment_name,
            model_name,
            model_path,
            ds_optimize,
            ds_zero,
            ds_config,
            mii_configs,
            mii_configs.port_number,
            "load balancer",
            ex_server_args=[f"--load-balancer {b64_config_str}"])

    def _launch_restful_gateway(self,
                                deployment_name,
                                model_name,
                                model_path,
                                ds_optimize,
                                ds_zero,
                                ds_config,
                                mii_configs,
                                port):
        return self._launch_server_process(deployment_name,
                                           model_name,
                                           model_path,
                                           ds_optimize,
                                           ds_zero,
                                           ds_config,
                                           mii_configs,
                                           port,
                                           "restful api gateway",
                                           ex_server_args=["--restful-gateway"])

    def _launch_server_process(self,
                               deployment_name,
                               model_name,
                               model_path,
                               ds_optimize,
                               ds_zero,
                               ds_config,
                               mii_configs,
                               port,
                               msg_server_type,
                               ds_launch_str=None,
                               ex_server_args=[]):
        launch_str = f"{sys.executable} -m mii.launch.multi_gpu_server"
        server_args_str = self._build_server_args(deployment_name,
                                                  model_name,
                                                  model_path,
                                                  ds_optimize,
                                                  ds_zero,
                                                  ds_config,
                                                  mii_configs,
                                                  port)
        server_args_str += f" " + \
            " ".join(ex_server_args) if ex_server_args else ""

        if ds_launch_str is None:
            cmd = f'{launch_str} {server_args_str}'.split(" ")
        else:
            cmd = f'{ds_launch_str} {launch_str} {server_args_str}'.split(" ")

        mii_env = os.environ.copy()
        mii_env["TRANSFORMERS_CACHE"] = model_path
        logger.info(f"{msg_server_type} server launch: {cmd}")
        return subprocess.Popen(cmd, env=mii_env)

    def _launch_deepspeed(self,
                          deployment_name,
                          model_name,
                          model_path,
                          ds_optimize,
                          ds_zero,
                          ds_config,
                          mii_configs,
                          hostfile,
                          host,
                          port,
                          master_port,
                          deploy_ranks):
        # use different hostfiles for replica instances
        # pass /dev/null when no replica is used
        worker_str = f"-H {hostfile} "
        # pin deepspeed launch to specific gpu id(s)
        included_gpus = f"{host}:{','.join(map(str, deploy_ranks))}"
        worker_str += f"-i {included_gpus} "

        # adjust torch dist port depending on rank, otherwise multi-replica deployments will conflict
        # assign different ports to replicas because they could be on the same host
        worker_str += f"--master_port {master_port}"

        ds_launch_str = f"deepspeed {worker_str} --master_addr localhost --no_ssh_check --no_local_rank --no_python"

        return self._launch_server_process(deployment_name,
                                           model_name,
                                           model_path,
                                           ds_optimize,
                                           ds_zero,
                                           ds_config,
                                           mii_configs,
                                           port,
                                           "MII server",
                                           ds_launch_str=ds_launch_str)

    def _initialize_service(self,
                            deployment_name,
                            model_name,
                            model_path,
                            ds_optimize,
                            ds_zero,
                            ds_config,
                            mii_configs,
                            lb_config):

        processes = []

        host_gpus = defaultdict(list)
        for repl_config in lb_config.replica_configs:
            host_gpus[repl_config.hostname].extend(repl_config.gpu_indices)

        # Start replica instances
        for i, repl_config in enumerate(lb_config.replica_configs):
            hostfile = tempfile.NamedTemporaryFile(delete=False)
            hostfile.write(
                f'{repl_config.hostname} slots={max(host_gpus[repl_config.hostname])+1}\n'
                .encode())
            processes.append(
                self._launch_deepspeed(
                    deployment_name,
                    model_name,
                    model_path,
                    ds_optimize,
                    ds_zero,
                    ds_config,
                    mii_configs,
                    hostfile.name,
                    repl_config.hostname,
                    repl_config.tensor_parallel_ports[0],
                    mii_configs.torch_dist_port + (100 * i) + repl_config.gpu_indices[0],
                    repl_config.gpu_indices))

            # start load balancer here.
            # we don't use deepspeed launcher for the load balancer because it does not need a GPU.
            # The deepspeed launcher determines the number of processes to launch based on GPUs available on the host or CUDA_VISIBLE_DEVICES,
            # and it is expected to assign one GPU to one process.
        processes.append(
            self._launch_load_balancer(deployment_name,
                                       model_name,
                                       model_path,
                                       ds_optimize,
                                       ds_zero,
                                       ds_config,
                                       mii_configs,
                                       lb_config))

        if mii_configs.enable_restful_api:
            # start rest api server
            processes.append(
                self._launch_restful_gateway(deployment_name,
                                             model_name,
                                             model_path,
                                             ds_optimize,
                                             ds_zero,
                                             ds_config,
                                             mii_configs,
                                             mii_configs.port_number))

        return processes


def _allocate_processes(hostfile_path, tensor_parallel, num_replicas):
    resource_pool = fetch_hostfile(hostfile_path)
    assert resource_pool is not None and len(
        resource_pool) > 0, f'No hosts found in {hostfile_path}'

    replica_pool = []
    allocated_num = 0
    for host, slots in resource_pool.items():
        available_on_host = slots
        while available_on_host >= tensor_parallel:
            if allocated_num >= num_replicas:
                break
            if slots < tensor_parallel:
                raise ValueError(
                    f'Host {host} has {slots} slot(s), but {tensor_parallel} slot(s) are required'
                )

            allocated_num_on_host = slots - available_on_host
            replica_pool.append(
                (host,
                 [
                     i for i in range(allocated_num_on_host,
                                      allocated_num_on_host + tensor_parallel)
                 ]))
            allocated_num += 1

            available_on_host -= tensor_parallel

    if allocated_num < num_replicas:
        raise ValueError(
            f'No sufficient GPUs for {num_replicas} replica(s), only {allocated_num} replica(s) can be deployed'
        )

    return replica_pool
