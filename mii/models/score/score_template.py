# flake8: noqa
'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import json
import torch
import mii
from mii.config import LoadBalancerConfig, ReplicaConfig
import time

model = None


def init():
    model_path = mii.utils.full_model_path(configs[mii.constants.MODEL_PATH_KEY])

    deployment_name = configs[mii.constants.DEPLOYMENT_NAME_KEY]
    model_name = configs[mii.constants.MODEL_NAME_KEY]
    task_name = configs[mii.constants.TASK_NAME_KEY]

    assert model_name is not None, "The model name should be set before calling init"
    assert task_name is not None, "The task name should be set before calling init"

    mii.MIIServer(deployment_name,
                  task_name,
                  model_name,
                  model_path,
                  ds_optimize=configs[mii.constants.ENABLE_DEEPSPEED_KEY],
                  ds_zero=configs[mii.constants.ENABLE_DEEPSPEED_ZERO_KEY],
                  ds_config=configs[mii.constants.DEEPSPEED_CONFIG_KEY],
                  mii_configs=configs[mii.constants.MII_CONFIGS_KEY],
                  lb_config=configs.get(mii.constants.LOAD_BALANCER_CONFIG_KEY,
                                        None))

    global model
    model = None

    # In AML deployments both the GRPC client and server are used in the same process
    if mii.utils.is_aml():
        model = mii.MIIClient(task_name,
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
