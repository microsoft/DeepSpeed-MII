'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import mii


model = None

def init():

    #TODO set the parallelism degree somehow. On the azure kubernetes server we can set the gpu core
    #but how do we do this in local deployment?
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    '''
        model_path: either MII_PATH or AML_MODEL_DIR depending on if its running with AML or not
        use_grpc_server: True if not running on AML.
        initialize_grpc_client: True only if running on AML.
    '''
    model_path, use_grpc_server, initialize_grpc_client = mii.setup_task()

    #if running in aml environment
    global model, configs

    model_name = configs['model_name']
    task = configs['task_name']

    assert model_name is not None, "The model name should be set before calling init"
    assert task is not None, "The task name should be set before calling init"
    
    model = mii.MIIServerClient(
        task,
        model_name,
        model_path,
        use_grpc_server=use_grpc_server,
        initialize_grpc_client=initialize_grpc_client)


def run(request):
    global model
    request_dict = json.loads(request)
    return model.query(request_dict)
