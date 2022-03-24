import os
import mii

generator = None


def init():

    #TODO set the parallelism degree somehow. On the azure kubernetes server we can set the gpu core
    #but how do we do this in local deployment?
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    '''
        model_path: either MII_PATH or AML_MODEL_DIR depending on if its running with AML or not
        use_grpc_server: True if not running on AML.
        initialize_grpc_client: True only if running on AML.
    '''
    model_path, use_grpc_server, initialize_grpc_client = mii.setup_generation_task()

    #if running in aml environment
    global generator
    generator = mii.MIIGenerationServerClient(
        'gpt2',
        model_path,
        use_grpc_server=use_grpc_server,
        initialize_grpc_client=initialize_grpc_client)


def run(request):
    global generator
    text = json.loads(request)
    return generator.query(text["query"])
