# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# flake8: noqa
import os
import json
import time
import torch

import mii

model = None


def init():
    global mii_config
    mii_config = mii.config.MIIConfig(**mii_config)

    # For AML deployments, we stand up multiple nginx server workers, one for
    # each replica. This is so that we can properly run multiple requests in
    # parallel on the different replicas. However, each worker will run this
    # generated score.py and try to stand up an entire MII deployment
    # (load-balancer, replicas, etc.). We want only one worker to spawn the
    # load-balancer and replicas. We take advantage of the nginx worker PIDs
    # being consecutive to achieve that here.
    start_server = True
    if mii.utils.is_aml() and (int(os.getpid()) % mii_config.replica_num != 0):
        start_server = False

    if start_server:
        mii.backend.MIIServer(mii_config)

    global model
    model = None

    # In AML deployments both the GRPC client and server are used in the same process
    if mii.utils.is_aml():
        model = mii.backend.MIIClient(mii_config=mii_config)


def run(request):
    global mii_config, model
    assert (
        model is not None
    ), "grpc client has not been setup when this model was created"

    request_dict = json.loads(request)

    query_dict = mii.utils.extract_query_dict(mii_config.task, request_dict)

    response = model.query(query_dict, **request_dict)

    time_taken = response.time_taken
    if not isinstance(response.response, str):
        response = [r for r in response.response]
    return json.dumps({"responses": response, "time": time_taken})


### Auto-generated config will be appended below at run-time
