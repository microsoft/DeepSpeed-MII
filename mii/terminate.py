# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import grpc

import mii


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

    generator.terminate()
    mii.client.terminate_restful_gateway(deployment_name)
