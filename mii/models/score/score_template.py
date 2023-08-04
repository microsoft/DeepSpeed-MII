# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# flake8: noqa
import os
import json
import torch
import mii
from mii.config import LoadBalancerConfig, ReplicaConfig, MIIConfig
import time

model = None


def init():
    mii_config = MIIConfig(**mii_config)
    mii.MIIServer(mii_config)

    global model
    model = None

    # In AML deployments both the GRPC client and server are used in the same process
    if mii.utils.is_aml():
        model = mii.MIIClient(mii_config.task_name,
                              mii_configs=configs[mii.constants.MII_CONFIGS_KEY])


def run(request):
    global model
    assert model is not None, "grpc client has not been setup when this model was created"

    request_dict = json.loads(request)

    query_dict = mii.utils.extract_query_dict(configs[mii.constants.TASK_NAME_KEY],
                                              request_dict)

    response = model.query(query_dict, **request_dict)

    time_taken = response.time_taken
    if not isinstance(response.response, str):
        response = [r for r in response.response]
    return json.dumps({'responses': response, 'time': time_taken})


### Auto-generated config will be appended below at run-time
