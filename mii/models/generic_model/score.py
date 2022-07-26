'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import json
import mii
import time

model = None


def init():
    model_path, use_grpc_server, initialize_grpc_client = mii.setup_task()

    aml_model_path = configs[mii.constants.MII_CONFIGS_KEY].get(
        mii.constants.AML_MODEL_PATH_KEY,
        None)
    if aml_model_path:
        model_path = os.path.join(model_path, aml_model_path)

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

    query_dict = mii.extract_query_dict(configs[mii.constants.TASK_NAME_KEY],
                                        request_dict)

    response = model.query(query_dict, **request_dict)

    if not isinstance(response.response, str):
        response = [r for r in response.response]
    return json.dumps({'responses': response})


### Auto-generated config will be appended below at run-time
