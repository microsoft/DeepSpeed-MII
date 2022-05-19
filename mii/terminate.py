import grpc
import psutil

import mii

def terminate_local_server(name):
    mii.utils.logger.info(f"Terminating server for {name}")
    deployment_name = name + '_deployment'
    generator = mii.mii_query_handle(name + "_deployment")
    try:
        generator.query({'query':None})
    except grpc.aio._call.AioRpcError as error:
        if error._code == grpc.StatusCode.UNAVAILABLE:
            mii.utils.logger.warn(f"Server for {name} not found")
        else:
            raise error
    except TypeError as error:
        pass

    mii_configs = mii.utils.import_score_file(deployment_name).configs[mii.constants.MII_CONFIGS_KEY]
    server_ports = [mii_configs['port_number']+i for i in range(mii_configs['tensor_parallel'])]
    for conn in psutil.net_connections():
        if conn.laddr.port in server_ports:
            p = psutil.Process(conn.pid)
            p.terminate()
