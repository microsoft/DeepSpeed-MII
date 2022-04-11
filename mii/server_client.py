'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import asyncio
import torch
import sys
import subprocess
import time
import grpc
import mii
from mii.utils import logger


def mii_query_handle(deployment_name):

    configs = mii.utils.import_score_file(deployment_name).configs

    task = configs[mii.constants.TASK_NAME_KEY]

    assert task is not None, "The task name should be set before calling init"

    return mii.MIIServerClient(task,
                               "na",
                               "na",
                               mii_configs=configs[mii.constants.MII_CONFIGS_KEY],
                               initialize_service=False,
                               initialize_grpc_client=True,
                               use_grpc_server=True)


class MIIServerClient():
    '''Initialize the model, setup the server and client for the model under model_path'''
    def __init__(self,
                 task_name,
                 model_name,
                 model_path,
                 ds_optimize=True,
                 mii_configs=mii.constants.MII_CONFIGS_DEFAULT,
                 initialize_service=True,
                 initialize_grpc_client=True,
                 use_grpc_server=False):

        self.task = mii.get_task(task_name)

        self.num_gpus = self._get_num_gpus(mii_configs)
        assert self.num_gpus > 0, "GPU count must be greater than 0"

        # This is true in two cases
        # i) If its multi-GPU
        # ii) It its a local deployment without AML
        self.use_grpc_server = True if (self.num_gpus > 1) else use_grpc_server
        self.initialize_service = initialize_service
        self.initialize_grpc_client = initialize_grpc_client

        self.port_number = mii_configs[mii.constants.PORT_NUMBER_KEY]

        if initialize_service and not self.use_grpc_server:
            self.model = None

        if self.initialize_service:
            self.process = self._initialize_service(model_name, model_path, ds_optimize)
            if self.use_grpc_server:
                self._wait_until_server_is_live()

        if self.initialize_grpc_client and self.use_grpc_server:
            self.stubs = []
            self.asyncio_loop = asyncio.get_event_loop()
            self._initialize_grpc_client()

    def _get_num_gpus(self, mii_configs):
        def get_tensor_parallel_gpus(mii_configs):
            TP_KEY = mii.constants.TENSOR_PARALLEL_KEY
            assert TP_KEY in mii_configs, "Must have tensor parallelism key in parallelism config"
            num_gpus = mii_configs[TP_KEY]

            assert torch.cuda.device_count() >= num_gpus, f"Available GPU count: {torch.cuda.device_count()} does not meet the required gpu count: {num_gpus}"
            return num_gpus

        # Only Tensor Parallelism supported for now
        return get_tensor_parallel_gpus(mii_configs)

    def _wait_until_server_is_live(self):
        sockets_open = False
        while not sockets_open:
            sockets_open = self._is_socket_open(self.port_number)
            process_alive = self._is_server_process_alive()
            if not process_alive:
                raise RuntimeError("server crashed for some reason, unable to proceed")
            time.sleep(4)
            logger.info("waiting for server to start...")
        logger.info(f"server has started on {self.port_number}")

    def _is_socket_open(self, port):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('0.0.0.0', port))
        sock.close()
        return result == 0

    def _is_server_process_alive(self):
        if self.process is None:
            return True
        try:
            self.process.wait(1)
        except subprocess.TimeoutExpired as err:
            # timeout means we're still running and all (probably) okay
            is_alive = True
        else:
            # no exception case
            is_alive = False
        return is_alive

    def _initialize_service(self, model_name, model_path, ds_optimize):
        process = None
        if not self.use_grpc_server:
            self.model = mii.load_models(mii.get_task_name(self.task), model_name, model_path, ds_optimize)
        else:
            if self._is_socket_open(self.port_number):
                raise RuntimeError(
                    f"Server is already running on port {self.port_number}, please shutdown to use different port."
                )

            ds_launch_str = f"deepspeed --num_gpus {self.num_gpus} --no_local_rank --no_python"
            launch_str = f"{sys.executable} -m mii.launch.multi_gpu_server"
            server_args_str = f"--task-name {mii.get_task_name(self.task)} --model {model_name} --model-path {model_path} --port {self.port_number}"
            server_args_str += " --ds-optimize" if ds_optimize else ""
            cmd = f'{ds_launch_str} {launch_str} {server_args_str}'.split(" ")
            print(cmd)
            process = subprocess.Popen(cmd)
            #TODO: do we need to hold onto this process handle for clean-up purposes?
        return process

    def _initialize_grpc_client(self):
        channels = []
        for i in range(self.num_gpus):
            channel = grpc.aio.insecure_channel(f'localhost:{self.port_number + i}')
            stub = mii.grpc_related.proto.modelresponse_pb2_grpc.ModelResponseStub(
                channel)
            channels.append(channel)
            self.stubs.append(stub)

    #runs task in parallel and return the result from the first task
    async def _query_in_tensor_parallel(self, request_string):
        responses = []
        for i in range(self.num_gpus):
            responses.append(
                self.asyncio_loop.create_task(
                    self._request_async_response(i,
                                                 request_string)))

        await responses[0]

        return responses[0]

    async def _request_async_response(self, stub_id, request_dict):
        if self.task == mii.Tasks.TEXT_GENERATION:
            response = await self.stubs[stub_id].GeneratorReply(
                mii.modelresponse_pb2.SingleStringRequest(request=request_dict['query']))

        elif self.task == mii.Tasks.TEXT_CLASSIFICATION:
            response = await self.stubs[stub_id].EntailmentReply(
                mii.modelresponse_pb2.SingleStringRequest(request=request_dict['query']))

        elif self.task == mii.Tasks.QUESTION_ANSWERING:
            response = await self.stubs[stub_id].QuestionAndAnswerReply(
                mii.modelresponse_pb2.QARequest(question=request_dict['question'],
                                                context=request_dict['context']))
        else:
            assert False, "unknown task"
        return response

    def _request_response(self, request_dict):
        start = time.time()
        if self.task == mii.Tasks.TEXT_GENERATION:
            response = self.model(request_dict['query'], do_sample=True, min_length=50)

        elif self.task == mii.Tasks.TEXT_CLASSIFICATION:
            response = self.model(request_dict['query'], return_all_scores=True)

        elif self.task == mii.Tasks.QUESTION_ANSWERING:
            response = self.model(question=request_dict['query'],
                                  context=request_dict['context'])
        else:
            raise NotSupportedError(f"task is not supported: {self.task}")
        end = time.time()
        return f"{response}" + f"\n Model Execution Time: {end-start} seconds"

    def query(self, request_dict):

        if not self.use_grpc_server:
            response = self._request_response(request_dict)
            generated_string = f"{response}"
        else:
            assert self.initialize_grpc_client, "grpc client has not been setup when this model was created"
            response = self.asyncio_loop.run_until_complete(
                self._query_in_tensor_parallel(request_dict))
            generated_string = response.result().response

        return generated_string
