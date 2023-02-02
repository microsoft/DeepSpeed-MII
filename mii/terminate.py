import grpc
import psutil
import os

import mii
from mii.constants import DeploymentType, DEPLOYMENT_TYPE_KEY, MII_CONFIGS_KEY
from mii.models.score import generated_score_path


def terminate(deployment_name):
    mii.utils.logger.info(f"Terminating server for {deployment_name}")
    generator = mii.mii_query_handle(deployment_name)
    try:
        generator.query({'query': ''})
    except grpc.aio._call.AioRpcError as error:
        if error._code == grpc.StatusCode.UNAVAILABLE:
            mii.utils.logger.warn(f"Server for {deployment_name} not found")
        else:
            pass
    except (KeyError, TypeError) as error:
        pass

    config = mii.utils.import_score_file(deployment_name).configs
    mii_configs = config[MII_CONFIGS_KEY]
    server_ports = []

    # If running the AML_LOCAL deployment, the azmlinfsrv process must be
    # shutdown first or it will respawn the MII server.
    if DeploymentType(config[DEPLOYMENT_TYPE_KEY]) == DeploymentType.AML_LOCAL:
        server_ports.append(mii_configs["aml_local_port"])

        score_path = generated_score_path(deployment_name, DeploymentType.AML_LOCAL)
        pid_file_path = os.path.join(os.path.dirname(score_path), "azmlinfsrv_pid")
        with open(pid_file_path, "r") as pid_file:
            pid = int(pid_file.read())
        p = psutil.Process(pid)
        p.terminate()

    server_ports.extend(
        [mii_configs['port_number'] + i for i in range(mii_configs['tensor_parallel'])])
    for conn in psutil.net_connections():
        if conn.laddr.port in server_ports:
            p = psutil.Process(conn.pid)
            p.terminate()
