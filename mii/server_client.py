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


def mii_query_handle(task, port_number=50050):
    return MIIServerClient(task,
                           "na",
                           "na",
                           initialize_service=False,
                           initialize_grpc_client=True,
                           use_grpc_server=True,
                           port_number=port_number)


class MIIServerClient():
    '''Initialize the model, setup the server and client for the model under model_path'''
    def __init__(self,
                 task_name,
                 model_name,
                 model_path,
                 initialize_service=True,
                 initialize_grpc_client=True,
                 use_grpc_server=False,
                 port_number=50050):

        self.task_name = task_name
        self.num_gpus = torch.cuda.device_count()
        assert self.num_gpus > 0, "No GPU detected"

        # This is true in two cases
        # i) If its multi-GPU
        # ii) It its a local deployment without AML
        self.use_grpc_server = True if (self.num_gpus > 1) else use_grpc_server
        self.initialize_service = initialize_service
        self.initialize_grpc_client = initialize_grpc_client

        self.port_number = port_number

        if initialize_service and not self.use_grpc_server:
            self.model = None

        if self.initialize_service:
            self.process = self._initialize_service(model_name, model_path)
            self._wait_until_server_is_live()

        if self.initialize_grpc_client and self.use_grpc_server:
            self.stubs = []
            self.asyncio_loop = asyncio.get_event_loop()
            self._initialize_grpc_client()

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

    def _initialize_service(self, model_name, model_path):
        process = None
        if not self.use_grpc_server:
            self.model = mii.load_model(model_name, model_path)
        else:
            if self._is_socket_open(self.port_number):
                raise RuntimeError(
                    f"Server is already running on port {self.port_number}, please shutdown to use different port."
                )
            #TODO we need to dump the log from these executions for debugging. Currently these are all lost
            ds_launch_str = f"deepspeed --num_gpus {self.num_gpus} --no_local_rank --no_python"
            launch_str = f"{sys.executable} -m mii.launch.multi_gpu_server"
            server_args_str = f"--task {self.task_name} --model {model_name} --model-path {model_path} --port {self.port_number}"
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
        if self.task_name in ['text-generation']:
            response = await self.stubs[stub_id].GeneratorReply(
                mii.modelresponse_pb2.SingleStringRequest(request=request_dict['query']))
        elif self.task_name in ['text-classification']:
            response = await self.stubs[stub_id].EntailmentReply(
                mii.modelresponse_pb2.SingleStringRequest(request=request_dict['query']))
        elif self.task_name in ['question-answering']:
            response = await self.stubs[stub_id].QuestionAndAnswerReply(
                mii.modelresponse_pb2.QARequest(question=request_dict['question'],
                                                context=request_dict['context']))
        else:
            assert False, "unknown task"
        return response

    def _request_response(self, request_dict):
        if self.task.GENERATION:
            response = self.model(request_dict['query'], do_sample=True, min_length=50)
        elif self.task.CLASSIFICATION:
            response = self.model(request_dict['query'], return_all_scores=True)
        elif self.task.QA:
            response = self.model(question=request_dict['query'],
                                  context=request_dict['context'])
        else:
            raise NotSupportedError(f"task is not supported: {self.task}")
        return response

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
