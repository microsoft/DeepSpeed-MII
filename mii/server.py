'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mii
from mii.utils import get_num_gpus, logger
from mii.config import ReplicaConfig


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

        if mii_configs.enable_load_balancing > 1 and mii_configs.hostfile is None:
            raise ValueError(
                "hostfile must be provided if enable_load_balancing == True")

        processes = self._initialize_service(model_name,
                                             model_path,
                                             ds_optimize,
                                             ds_zero,
                                             ds_config,
                                             mii_configs,
                                             lb_config)
        deployment = lb_config.replica_configs if mii_configs.enable_load_balancing else [
            ReplicaConfig(hostname='localhost',
                          tensor_parallel_ports=[mii_configs.port_number],
                          torch_dist_port=mii_configs.torch_dist_port)
        ]
        self._wait_until_server_is_live(processes, deployment)

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
                           model_name,
                           model_path,
                           ds_optimize,
                           ds_zero,
                           ds_config,
                           mii_configs,
                           port):
        # serialize mii config
        b64_config_str = config_to_b64_str(mii_configs)

        server_args_str = f"--task-name {mii.utils.get_task_name(self.task)} --model {model_name} --model-path {model_path} --port {port}"
        server_args_str += " --ds-optimize" if ds_optimize else ""

        # XXX: fetch model provider based on model name in a more general way
        if model_name == "gpt-neox":
            provider = mii.constants.MODEL_PROVIDER_NAME_EA
        elif ("bigscience/bloom" == model_name) or ("microsoft/bloom" in model_name):
            provider = mii.constants.MODEL_PROVIDER_NAME_HF_LLM
        elif self.task == mii.Tasks.TEXT2IMG:
            provider = mii.constants.MODEL_PROVIDER_NAME_DIFFUSERS
        else:
            provider = mii.constants.MODEL_PROVIDER_NAME_HF
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
                                model_name,
                                model_path,
                                ds_optimize,
                                ds_zero,
                                ds_config,
                                mii_configs,
                                port):
        return self._launch_server_process(model_name,
                                           model_path,
                                           ds_optimize,
                                           ds_zero,
                                           ds_config,
                                           mii_configs,
                                           port,
                                           "restful api gateway",
                                           ex_server_args=["--restful-gateway"])

    def _launch_server_process(self,
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
        server_args_str = self._build_server_args(model_name,
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

        ds_launch_str = f"deepspeed {worker_str} --no_local_rank --no_python"

        return self._launch_server_process(model_name,
                                           model_path,
                                           ds_optimize,
                                           ds_zero,
                                           ds_config,
                                           mii_configs,
                                           port,
                                           "MII server",
                                           ds_launch_str=ds_launch_str)

    def _initialize_service(self,
                            model_name,
                            model_path,
                            ds_optimize,
                            ds_zero,
                            ds_config,
                            mii_configs,
                            lb_config):

        processes = []
        if mii_configs.enable_load_balancing:

            # Start replica instances
            for i, repl_config in enumerate(lb_config.replica_configs):
                hostfile = tempfile.NamedTemporaryFile(delete=False)
                hostfile.write(
                    f'{repl_config.hostname} slots={mii_configs.replica_num}\n'.encode())
                processes.append(
                    self._launch_deepspeed(
                        model_name,
                        model_path,
                        ds_optimize,
                        ds_zero,
                        ds_config,
                        mii_configs,
                        hostfile.name,
                        repl_config.hostname,
                        repl_config.tensor_parallel_ports[0],
                        mii_configs.torch_dist_port + (100 * i) +
                        repl_config.gpu_indices[0],
                        repl_config.gpu_indices))

            # start load balancer here.
            # we don't use deepspeed launcher for the load balancer because it does not need a GPU.
            # The deepspeed launcher determines the number of processes to launch based on GPUs available on the host or CUDA_VISIBLE_DEVICES,
            # and it is expected to assign one GPU to one process.
            processes.append(
                self._launch_load_balancer(model_name,
                                           model_path,
                                           ds_optimize,
                                           ds_zero,
                                           ds_config,
                                           mii_configs,
                                           lb_config))

            return processes
        else:
            if self._is_socket_open("localhost", self.port_number):
                raise RuntimeError(
                    f"Server is already running on port {self.port_number}, please shutdown or use different port."
                )

            processes.append(
                self._launch_deepspeed(model_name,
                                       model_path,
                                       ds_optimize,
                                       ds_zero,
                                       ds_config,
                                       mii_configs,
                                       '/dev/null',
                                       'localhost',
                                       mii_configs.port_number,
                                       mii_configs.torch_dist_port,
                                       mii_configs.deploy_rank))
        return processes
