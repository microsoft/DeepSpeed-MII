# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import grpc

import mii


def terminate(deployment_tag):
    mii.utils.logger.info(f"Terminating server for {deployment_tag}")
    generator = mii.mii_query_handle(deployment_tag)
    if (deployment_tag in mii.non_persistent_models):
        generator.terminate()
        return
    try:
        generator.query({'query': ''}, mii.constants.MII_TERMINATE_DEP_NAME)
    except grpc.aio._call.AioRpcError as error:
        if error._code == grpc.StatusCode.UNAVAILABLE:
            mii.utils.logger.warn(f"Server for {deployment_tag} not found")
        else:
            pass
    except (KeyError, TypeError) as error:
        pass

    generator.terminate()
    mii.client.terminate_restful_gateway(deployment_tag)
