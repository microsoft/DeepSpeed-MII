'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import asyncio
import torch
import sys
import subprocess
import time
import grpc
import os
import json
from pathlib import Path
import mii
import base64
from mii.utils import logger, kwarg_dict_to_proto
from mii.grpc_related.proto import modelresponse_pb2, modelresponse_pb2_grpc


def mii_query_handle(deployment_name):
    """Get a query handle for a local deployment:

        mii/examples/local/gpt2-query-example.py
        mii/examples/local/roberta-qa-query-example.py


    Arguments:
        deployment_name: Name of the deployment. Used as an identifier for posting queries for ``LOCAL`` deployment.

    Returns:
        query_handle: A query handle with a single method `.query(request_dictionary)` using which queries can be sent to the model.

    """

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
                 ds_zero=False,
                 ds_config=None,
                 mii_configs={},
                 initialize_service=True,
                 initialize_grpc_client=True,
                 use_grpc_server=False):

        mii_configs = mii.config.MIIConfig(**mii_configs)

        self.task = mii.utils.get_task(task_name)

        self.num_gpus = self._get_num_gpus(mii_configs)
        assert self.num_gpus > 0, "GPU count must be greater than 0"

        # This is true in two cases
        # i) If its multi-GPU
        # ii) It is a local deployment
        self.use_grpc_server = True if (self.num_gpus > 1) else use_grpc_server
        self.initialize_service = initialize_service
        self.initialize_grpc_client = initialize_grpc_client

        self.port_number = mii_configs.port_number

        if initialize_service and not self.use_grpc_server:
            self.model = None

        if self.initialize_service:
            self.process = self._initialize_service(model_name,
                                                    model_path,
                                                    ds_optimize,
                                                    ds_zero,
                                                    ds_config,
                                                    mii_configs)
            if self.use_grpc_server:
                self._wait_until_server_is_live()

        if self.initialize_grpc_client and self.use_grpc_server:
            self.stubs = []
            self.asyncio_loop = asyncio.get_event_loop()
            self._initialize_grpc_client()

    def _get_num_gpus(self, mii_configs):
        def get_tensor_parallel_gpus(mii_configs):
            num_gpus = mii_configs.tensor_parallel

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

    def _initialize_service(self,
                            model_name,
                            model_path,
                            ds_optimize,
                            ds_zero,
                            ds_config,
                            mii_configs):
        process = None
        if not self.use_grpc_server:
            self.model = mii.models.load_models(task_name=mii.utils.get_task_name(
                self.task),
                                                model_name=model_name,
                                                model_path=model_path,
                                                ds_optimize=ds_optimize,
                                                ds_zero=ds_zero,
                                                ds_config_path=ds_config,
                                                mii_config=mii_configs)
        else:
            if self._is_socket_open(self.port_number):
                raise RuntimeError(
                    f"Server is already running on port {self.port_number}, please shutdown or use different port."
                )

            # serialize mii config
            # convert json str -> bytes
            json_bytes = mii_configs.json().encode()
            # base64 encoded bytes
            b64_config_bytes = base64.urlsafe_b64encode(json_bytes)
            # bytes -> str
            b64_config_str = b64_config_bytes.decode()

            #TODO: will need worker hostfile support here for multi-node launching, this force ignores a /job/hostfile
            #      if one exists which is not compatible when passing localhost as a hostname.
            worker_str = "-H /dev/null "
            # pin deepspeed launch to specific gpu id(s)
            worker_str += f"-i localhost:{','.join(map(str, mii_configs.deploy_rank))} "
            # adjust torch dist port depending on rank, otherwise multi-replica deployments will conflict
            worker_str += f"--master_port {mii_configs.torch_dist_port + mii_configs.deploy_rank[0]}"

            ds_launch_str = f"deepspeed {worker_str} --no_local_rank --no_python"
            launch_str = f"{sys.executable} -m mii.launch.multi_gpu_server"
            server_args_str = f"--task-name {mii.utils.get_task_name(self.task)} --model {model_name} --model-path {model_path} --port {self.port_number}"
            server_args_str += " --ds-optimize" if ds_optimize else ""

            #XXX: fetch model provider based on model name in a more general way
            if model_name == "gpt-neox":
                provider = mii.constants.MODEL_PROVIDER_NAME_EA
            elif ("bigscience/bloom" == model_name) or ("microsoft/bloom" in model_name):
                provider = mii.constants.MODEL_PROVIDER_NAME_HF_LLM
            elif self.task == mii.Tasks.TEXT2IMG:
                provider = mii.constants.MODEL_PROVIDER_NAME_DIFFUSERS
            else:
                provider = mii.constants.MODEL_PROVIDER_NAME_HF
            server_args_str += f" --provider {provider}"

            server_args_str += f" --config {b64_config_str}"
            server_args_str += " --ds-zero" if ds_zero else ""
            if ds_zero and ds_config is not None:
                if isinstance(ds_config, dict):

                    def create_config_from_dict(tmpdir, config_dict):
                        if not os.path.exists(tmpdir):
                            os.makedirs(tmpdir)
                        config_path = os.path.join(tmpdir, 'temp_config.json')
                        with open(config_path, 'w') as fd:
                            json.dump(config_dict, fd)
                        return config_path

                    model_dir = Path(model_path).parent.resolve()
                    ds_config_path = create_config_from_dict(model_dir, ds_config)
                elif isinstance(ds_config, str):
                    ds_config_path = ds_config
                else:
                    raise ValueError(
                        f"Expected a string path to an existing deepspeed config, or a dictionary. Received: {ds_config}"
                    )
                server_args_str += f" --ds-config {ds_config_path}"
            cmd = f'{ds_launch_str} {launch_str} {server_args_str}'.split(" ")
            printable_config = f"task-name {mii.utils.get_task_name(self.task)} model {model_name} model-path {model_path} port {self.port_number} provider {provider}"
            logger.info(f"MII using multi-gpu deepspeed launcher:\n" +
                        self.print_helper(printable_config))
            mii_env = os.environ.copy()
            mii_env["TRANSFORMERS_CACHE"] = model_path
            process = subprocess.Popen(cmd, env=mii_env)
        return process

    def print_helper(self, args):
        # convert to list
        args = args.split(" ")
        # convert to dict
        dct = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
        printable_string = ""
        printable_string += " " + "-" * 60 + "\n"
        for k, v in dct.items():
            dots = "." * (29 - len(k))
            printable_string += f" {k} {dots} {v} \n"
        printable_string += " " + "-" * 60
        return printable_string

    def _initialize_grpc_client(self):
        channels = []
        for i in range(self.num_gpus):
            channel = grpc.aio.insecure_channel(f'localhost:{self.port_number + i}')
            stub = modelresponse_pb2_grpc.ModelResponseStub(channel)
            channels.append(channel)
            self.stubs.append(stub)

    #runs task in parallel and return the result from the first task
    async def _query_in_tensor_parallel(self, request_string, query_kwargs):
        responses = []
        for i in range(self.num_gpus):
            responses.append(
                self.asyncio_loop.create_task(
                    self._request_async_response(i,
                                                 request_string,
                                                 query_kwargs)))

        await responses[0]

        return responses[0]

    async def _request_async_response(self, stub_id, request_dict, query_kwargs):
        proto_kwargs = kwarg_dict_to_proto(query_kwargs)
        if self.task == mii.Tasks.TEXT_GENERATION:
            # convert to batch of queries if they are not already
            if not isinstance(request_dict['query'], list):
                request_dict['query'] = [request_dict['query']]
            req = modelresponse_pb2.MultiStringRequest(request=request_dict['query'],
                                                       query_kwargs=proto_kwargs)
            response = await self.stubs[stub_id].GeneratorReply(req)

        elif self.task == mii.Tasks.TEXT_CLASSIFICATION:
            response = await self.stubs[stub_id].ClassificationReply(
                modelresponse_pb2.SingleStringRequest(request=request_dict['query'],
                                                      query_kwargs=proto_kwargs))

        elif self.task == mii.Tasks.QUESTION_ANSWERING:
            response = await self.stubs[stub_id].QuestionAndAnswerReply(
                modelresponse_pb2.QARequest(question=request_dict['question'],
                                            context=request_dict['context'],
                                            query_kwargs=proto_kwargs))
        elif self.task == mii.Tasks.FILL_MASK:
            response = await self.stubs[stub_id].FillMaskReply(
                modelresponse_pb2.SingleStringRequest(request=request_dict['query'],
                                                      query_kwargs=proto_kwargs))

        elif self.task == mii.Tasks.TOKEN_CLASSIFICATION:
            response = await self.stubs[stub_id].TokenClassificationReply(
                modelresponse_pb2.SingleStringRequest(request=request_dict['query'],
                                                      query_kwargs=proto_kwargs))

        elif self.task == mii.Tasks.CONVERSATIONAL:
            response = await self.stubs[stub_id].ConversationalReply(
                modelresponse_pb2.ConversationRequest(
                    text=request_dict['text'],
                    conversation_id=request_dict['conversation_id']
                    if 'conversation_id' in request_dict else None,
                    past_user_inputs=request_dict['past_user_inputs'],
                    generated_responses=request_dict['generated_responses'],
                    query_kwargs=proto_kwargs))

        elif self.task == mii.Tasks.TEXT2IMG:
            # convert to batch of queries if they are not already
            if not isinstance(request_dict['query'], list):
                request_dict['query'] = [request_dict['query']]
            req = modelresponse_pb2.MultiStringRequest(request=request_dict['query'],
                                                       query_kwargs=proto_kwargs)
            response = await self.stubs[stub_id].Txt2ImgReply(req)

        else:
            raise ValueError(f"unknown task: {self.task}")
        return response

    def _request_response(self, request_dict, query_kwargs):
        start = time.time()
        if self.task == mii.Tasks.TEXT_GENERATION:
            response = self.model(request_dict['query'], **query_kwargs)

        elif self.task == mii.Tasks.TEXT_CLASSIFICATION:
            response = self.model(request_dict['query'], **query_kwargs)

        elif self.task == mii.Tasks.QUESTION_ANSWERING:
            response = self.model(question=request_dict['query'],
                                  context=request_dict['context'],
                                  **query_kwargs)

        elif self.task == mii.Tasks.FILL_MASK:
            response = self.model(request_dict['query'], **query_kwargs)

        elif self.task == mii.Tasks.TOKEN_CLASSIFICATION:
            response = self.model(request_dict['query'], **query_kwargs)

        elif self.task == mii.Tasks.CONVERSATIONAL:
            response = self.model(["", request_dict['query']], **query_kwargs)

        elif self.task == mii.Tasks.TEXT2IMG:
            response = self.model(request_dict['query'], **query_kwargs)

        else:
            raise NotImplementedError(f"task is not supported: {self.task}")
        end = time.time()
        return f"{response}" + f"\n Model Execution Time: {end-start} seconds"

    def query(self, request_dict, **query_kwargs):
        """Query a local deployment:

            mii/examples/local/gpt2-query-example.py
            mii/examples/local/roberta-qa-query-example.py

        Arguments:
            request_dict: A task specific request dictionary consistinging of the inputs to the models
            query_kwargs: additional query parameters for the model

        Returns:
            response: Response of the model
        """
        if not self.use_grpc_server:
            response = self._request_response(request_dict, query_kwargs)
            ret = f"{response}"
        else:
            assert self.initialize_grpc_client, "grpc client has not been setup when this model was created"
            response = self.asyncio_loop.run_until_complete(
                self._query_in_tensor_parallel(request_dict,
                                               query_kwargs))
            ret = response.result()
        return ret
