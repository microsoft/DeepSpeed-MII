# flake8: noqa
'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import json
import mii
import time

model = None


def init():
    # In AML deployments both the GRPC client and server are used in the same process
    initialize_grpc_client = mii.utils.is_aml()

    # XXX: Always run grpc server, originally was "not is_aml()"
    use_grpc_server = True

    model_path = mii.utils.full_model_path(configs[mii.constants.MODEL_PATH_KEY])

    model_name = configs[mii.constants.MODEL_NAME_KEY]
    task = configs[mii.constants.TASK_NAME_KEY]

    assert model_name is not None, "The model name should be set before calling init"
    assert task is not None, "The task name should be set before calling init"

    global model
    model = mii.MIIServerClient(task,
                                model_name,
                                model_path,
                                ds_optimize=configs[mii.constants.ENABLE_DEEPSPEED_KEY],
                                ds_zero=configs[mii.constants.ENABLE_DEEPSPEED_ZERO_KEY],
                                ds_config=configs[mii.constants.DEEPSPEED_CONFIG_KEY],
                                mii_configs=configs[mii.constants.MII_CONFIGS_KEY],
                                use_grpc_server=use_grpc_server,
                                initialize_grpc_client=initialize_grpc_client)


def run(request):
    global model
    request_dict = json.loads(request)

    query_dict = mii.utils.extract_query_dict(configs[mii.constants.TASK_NAME_KEY],
                                              request_dict)

    response = model.query(query_dict, **request_dict)

    time_taken = response.time_taken
    if not isinstance(response.response, str):
        response = [r for r in response.response]
    return json.dumps({'responses': response, 'time': time_taken})


### Auto-generated config will be appended below at run-time
