# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import base64
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from deepspeed.accelerator import get_accelerator

from mii.legacy.utils import get_num_gpus
from mii.legacy.logging import logger


def config_to_b64_str(config):
    # convert json str -> bytes
    json_bytes = config.json().encode()
    # base64 encoded bytes
    b64_config_bytes = base64.urlsafe_b64encode(json_bytes)
    # bytes -> str
    return b64_config_bytes.decode()


class MIIServer:
    """Initialize the model, setup the server for the model under model_path"""
    def __init__(self, mii_config):

        self.task = mii_config.model_config.task
        self.num_gpus = get_num_gpus(mii_config)
        assert self.num_gpus > 0, "GPU count must be greater than 0"

        self.port_number = mii_config.port_number

        if not os.path.isfile(mii_config.hostfile):
            logger.info(f"Hostfile {mii_config.hostfile} not found, creating hostfile.")
            num_gpu = get_accelerator().device_count()
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                temp_file.write(f"localhost slots={num_gpu}")
            mii_config.hostfile = temp_file.name

        mii_config.generate_replica_configs()

        processes = self._initialize_service(mii_config)
        self._wait_until_server_is_live(processes,
                                        mii_config.model_config.replica_configs)

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

    def _launch_server_process(self,
                               model_config,
                               msg_server_type,
                               ds_launch_str="",
                               server_args=None):
        launch_str = f"{sys.executable} -m mii.legacy.launch.multi_gpu_server"
        b64_config_str = config_to_b64_str(model_config)
        if server_args is None:
            server_args = []
        server_args.append(f"--model-config {b64_config_str}")
        server_args_str = " ".join(server_args)
        cmd = f"{ds_launch_str} {launch_str} {server_args_str}".strip().split(" ")

        mii_env = os.environ.copy()
        mii_env["TRANSFORMERS_CACHE"] = model_config.model_path
        logger.info(f"{msg_server_type} server launch: {cmd}")
        return subprocess.Popen(cmd, env=mii_env)

    def _generate_ds_launch_str(self, replica_config, hostfile):
        # use different hostfiles for replica instances
        # pass /dev/null when no replica is used
        worker_str = f"-H {hostfile} "
        # pin deepspeed launch to specific gpu id(s)
        included_gpus = f"{replica_config.hostname}:{','.join(map(str, replica_config.gpu_indices))}"
        worker_str += f"-i {included_gpus} "

        # adjust torch dist port depending on rank, otherwise multi-replica deployments will conflict
        # assign different ports to replicas because they could be on the same host
        worker_str += f"--master_port {replica_config.torch_dist_port}"

        ds_launch_str = f"deepspeed {worker_str} --master_addr localhost --no_ssh_check --no_local_rank --no_python"

        return ds_launch_str

    def _initialize_service(self, mii_config):
        processes = []
        server_args = [
            f"--deployment-name {mii_config.deployment_name}",
            f"--load-balancer-port {mii_config.port_number}",
            f"--restful-gateway-port {mii_config.restful_api_port}",
        ]

        host_gpus = defaultdict(list)
        for repl_config in mii_config.model_config.replica_configs:
            host_gpus[repl_config.hostname].extend(repl_config.gpu_indices)

        # Start replica instances
        for repl_config in mii_config.model_config.replica_configs:
            hostfile = tempfile.NamedTemporaryFile(delete=False)
            hostfile.write(
                f"{repl_config.hostname} slots={max(host_gpus[repl_config.hostname])+1}\n"
                .encode())
            ds_launch_str = self._generate_ds_launch_str(repl_config, hostfile.name)
            processes.append(
                self._launch_server_process(
                    mii_config.model_config,
                    "MII server",
                    ds_launch_str=ds_launch_str,
                    server_args=server_args +
                    [f"--server-port {repl_config.tensor_parallel_ports[0]}"],
                ))
            # start load balancer here. We don't use deepspeed launcher for the
            # load balancer because it does not need a GPU. The deepspeed
            # launcher determines the number of processes to launch based on
            # GPUs available on the host or CUDA_VISIBLE_DEVICES, and it is
            # expected to assign one GPU to one process.
        processes.append(
            self._launch_server_process(
                mii_config.model_config,
                "load balancer",
                server_args=server_args + ["--load-balancer"],
            ))

        if mii_config.enable_restful_api:
            processes.append(
                self._launch_server_process(
                    mii_config.model_config,
                    "restful api gateway",
                    server_args=server_args + ["--restful-gateway"],
                ))

        return processes
