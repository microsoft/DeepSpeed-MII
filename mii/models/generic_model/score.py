'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import json
import mii
import time

model = None


def init():

    #TODO set the parallelism degree somehow. On the azure kubernetes server we can set the gpu core
    #but how do we do this in local deployment?
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    '''
        model_path: either MII_PATH or AML_MODEL_DIR depending on if its running with AML or not
        use_grpc_server: True if not running on AML.
        initialize_grpc_client: True only if running on AML.
    '''
    model_path, use_grpc_server, initialize_grpc_client = mii.setup_task()

    #if running in aml environment
    global model, configs

    model_name = configs[mii.constants.MODEL_NAME_KEY]
    task = configs[mii.constants.TASK_NAME_KEY]

    assert model_name is not None, "The model name should be set before calling init"
    assert task is not None, "The task name should be set before calling init"

    model = mii.MIIServerClient(task,
                                model_name,
                                model_path,
                                ds_optimize = configs[mii.constants.ENABLE_DEEPSPEED_KEY],
                                mii_configs = configs[mii.constants.MII_CONFIGS_KEY],
                                use_grpc_server=use_grpc_server,
                                initialize_grpc_client=initialize_grpc_client)


def run(request):
    start = time.time()
    global model
    request_dict = json.loads(request)

    response = model.query(request_dict)
    end = time.time()
    response += f"\n Query Run Time: {end-start} secs"
    return response


### Auto-generated config will be appended below at run-time
