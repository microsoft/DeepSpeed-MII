# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import asyncio
import grpc
import requests
import mii
from mii.utils import get_task
from mii.grpc_related.proto import modelresponse_pb2, modelresponse_pb2_grpc
from mii.constants import GRPC_MAX_MSG_SIZE, Tasks, DeploymentType
from mii.method_table import GRPC_METHOD_TABLE
from mii.deployment import allocate_processes, create_score_file
from mii.config import DeploymentConfig


def _get_deployment_configs(deployment_tag):
    deployments = []
    configs = mii.utils.import_score_file(deployment_tag).configs
    for deployment in configs:
        if not isinstance(configs[deployment], dict):
            continue
        configs[deployment][mii.constants.DEPLOYED_KEY] = True
        deployments.append(configs[deployment])
    return deployments


def mii_query_handle(deployment_tag):
    """Get a query handle for a local deployment:

        mii/examples/local/gpt2-query-example.py
        mii/examples/local/roberta-qa-query-example.py

    Arguments:
        deployment_name: Name of the deployment. Used as an identifier for posting queries for ``LOCAL`` deployment.

    Returns:
        query_handle: A query handle with a single method `.query(request_dictionary)` using which queries can be sent to the model.
    """

    if deployment_tag in mii.non_persistent_models:
        inference_pipeline, task = mii.non_persistent_models[deployment_tag]
        return MIINonPersistentClient(task, deployment_tag)

    deployments = _get_deployment_configs(deployment_tag)
    if len(deployments) > 0:
        mii_configs_dict = deployments[0][mii.constants.MII_CONFIGS_KEY]
        mii_configs = mii.config.MIIConfig(**mii_configs_dict)

    return MIIClient(deployments, "localhost", mii_configs.port_number)


def create_channel(host, port):
    return grpc.aio.insecure_channel(f'{host}:{port}',
                                     options=[('grpc.max_send_message_length',
                                               GRPC_MAX_MSG_SIZE),
                                              ('grpc.max_receive_message_length',
                                               GRPC_MAX_MSG_SIZE)])


class MIIClient():
    """
    Client to send queries to a single endpoint.
    """
    def __init__(self, deployments, host, port):
        self.asyncio_loop = asyncio.get_event_loop()
        channel = create_channel(host, port)
        self.stub = modelresponse_pb2_grpc.DeploymentManagementStub(channel)
        #self.task = get_task(task_name)
        self.deployments = deployments

    def _get_deployment_task(self, deployment_name=None):
        task = None
        if deployment_name is None:  #mii.terminate() or single model
            assert len(self.deployments) == 1, "Must pass deployment_name to query when using multiple deployments"
            deployment_name = self.deployments[0][mii.constants.DEPLOYMENT_NAME_KEY]
            task = get_task(self.deployments[0][mii.constants.TASK_NAME_KEY])
        else:
            for deployment in self.deployments:
                if deployment[mii.constants.DEPLOYMENT_NAME_KEY] == deployment_name:
                    task = get_task(deployment[mii.constants.TASK_NAME_KEY])
                    return deployment_name, task
            assert False, f"{deployment_name} not found in list of deployments"
        return deployment_name, task

    async def _request_async_response(self, request_dict, task, **query_kwargs):
        if task not in GRPC_METHOD_TABLE:
            raise ValueError(f"unknown task: {task}")

        task_methods = GRPC_METHOD_TABLE[task]
        proto_request = task_methods.pack_request_to_proto(request_dict, **query_kwargs)
        proto_response = await getattr(self.stub, task_methods.method)(proto_request)
        return task_methods.unpack_response_from_proto(proto_response)

    def query(self, request_dict, **query_kwargs):
        deployment_name = request_dict.get('deployment_name')
        deployment_name, task = self._get_deployment_task(deployment_name)
        request_dict['deployment_name'] = deployment_name
        return self.asyncio_loop.run_until_complete(
            self._request_async_response(request_dict,
                                         task,
                                         **query_kwargs))

    async def terminate_async(self):
        await self.stub.Terminate(
            modelresponse_pb2.google_dot_protobuf_dot_empty__pb2.Empty())

    def terminate(self):
        self.asyncio_loop.run_until_complete(self.terminate_async())

    async def create_session_async(self, session_id):
        return await self.stub.CreateSession(
            modelresponse_pb2.SessionID(session_id=session_id))

    def create_session(self, session_id, deployment_name=None):
        if len(self.deployments > 1):
            assert deployment_name is not None, "Deployment name must be passed in to create session when there are multiple models"
        deployment_name, task = self._get_deployment_task(deployment_name)
        assert task == Tasks.TEXT_GENERATION, f"Session creation only available for task '{Tasks.TEXT_GENERATION}'."
        return self.asyncio_loop.run_until_complete(
            self.create_session_async(session_id))

    async def destroy_session_async(self, session_id):
        await self.stub.DestroySession(modelresponse_pb2.SessionID(session_id=session_id)
                                       )

    def destroy_session(self, session_id, deployment_name=None):
        if len(self.deployments > 1):
            assert deployment_name is not None, "Deployment name must be passed in to destroy session when there are multiple models"
        deployment_name, task = self._get_deployment_task(deployment_name)
        assert task == Tasks.TEXT_GENERATION, f"Session deletion only available for task '{Tasks.TEXT_GENERATION}'."
        self.asyncio_loop.run_until_complete(self.destroy_session_async(session_id))

    async def add_models_async(self, request=None):
        await getattr(self.stub, "AddDeployment")(modelresponse_pb2.google_dot_protobuf_dot_empty__pb2.Empty())

    def add_models(self,
                   task=None,
                   model=None,
                   deployment_name=None,
                   enable_deepspeed=True,
                   enable_zero=False,
                   ds_config=None,
                   mii_config={},
                   deployment_tag=None,
                   deployments=[],
                   deployment_type=DeploymentType.LOCAL,
                   model_path=None,
                   version=1):
        if not deployments:
            assert all((model, task, deployment_name)), "model, task, and deployment name must be set to deploy singular model"
            deployments = [
                DeploymentConfig(deployment_name=deployment_name,
                             task=task,
                             model=model,
                             enable_deepspeed=enable_deepspeed,
                             enable_zero=enable_zero,
                             GPU_index_map=None,
                             mii_config=mii.config.MIIConfig(**mii_config),
                             ds_config=ds_config,
                             version=version,
                             deployed=False)
            ]

        
        deployment_tag = mii.deployment_tag
        lb_config = allocate_processes(deployments)
        if mii.lb_config is not None:
            mii.lb_config.replica_configs.extend(lb_config.replica_configs)
        else:
            mii.lb_config = lb_config
        self.deployments.extend(deployments)
        if mii.model_path is None and deployment_type == DeploymentType.LOCAL:
            mii.model_path = MII_MODEL_PATH_DEFAULT
        elif mii.model_path is None and deployment_type == DeploymentType.AML:
            model_path = "model"
        deps = []
        for deployment in self.deployments:
             data = {
                'deployment_name': deployment[mii.constants.DEPLOYMENT_NAME_KEY],
                'task': deployment[mii.constants.TASK_NAME_KEY],
                'model': deployment[mii.constants.MODEL_NAME_KEY],
                'enable_deepspeed': deployment[mii.constants.ENABLE_DEEPSPEED_KEY],
                'enable_zero': deployment[mii.constants.ENABLE_DEEPSPEED_ZERO_KEY],
                'GPU_index_map': None,
                'mii_config': deployment[mii.constants.MII_CONFIGS_KEY],
                'ds_config': deployment[mii.constants.DEEPSPEED_CONFIG_KEY],
                'version': 1
                'deployed' deployment[mii.constants.DEPLOYED_KEY]
            }
             
        create_score_file(deployment_tag=deployment_tag, deployment_type=mii.deployment_type, deployments=self.deployments, model_path=mii.model_path, lb_config=mii.lb_config)
        if mii.deployment_type == DeploymentType.Local:
            mii.utils.import_score_file(deployment_tag).init()
        
        self.asyncio_loop.run_until_complete(self.add_models_async())
class MIITensorParallelClient():
    """
    Client to send queries to multiple endpoints in parallel.
    This is used to call multiple servers deployed for tensor parallelism.
    """
    def __init__(self, task_name, host, ports):
        self.task = get_task(task_name)
        self.clients = [MIIClient(task_name, host, port) for port in ports]
        self.asyncio_loop = asyncio.get_event_loop()

    # runs task in parallel and return the result from the first task
    async def _query_in_tensor_parallel(self, request_string, query_kwargs):
        responses = []
        for client in self.clients:
            responses.append(
                self.asyncio_loop.create_task(
                    client._request_async_response(request_string,
                                                   **query_kwargs)))

        await responses[0]
        return responses[0]

    def query(self, request_dict, **query_kwargs):
        """Query a local deployment:

            mii/examples/local/gpt2-query-example.py
            mii/examples/local/roberta-qa-query-example.py

        Arguments:
            request_dict: A task specific request dictionary consisting of the inputs to the models
            query_kwargs: additional query parameters for the model

        Returns:
            response: Response of the model
        """
        response = self.asyncio_loop.run_until_complete(
            self._query_in_tensor_parallel(request_dict,
                                           query_kwargs))
        ret = response.result()
        return ret

    def terminate(self):
        """Terminates the deployment"""
        for client in self.clients:
            client.terminate()

    def create_session(self, session_id):
        for client in self.clients:
            client.create_session(session_id)

    def destroy_session(self, session_id):
        for client in self.clients:
            client.destroy_session(session_id)


class MIINonPersistentClient():
    def __init__(self, task, deployment_name):
        self.task = get_task(task)
        self.deployment_name = deployment_name

    def query(self, request_dict, **query_kwargs):
        assert self.deployment_name in mii.non_persistent_models, f"deployment: {self.deployment_name} not found"
        task_methods = GRPC_METHOD_TABLE[self.task]
        inference_pipeline = mii.non_persistent_models[self.deployment_name][0]

        if self.task == Tasks.QUESTION_ANSWERING:
            if 'question' not in request_dict or 'context' not in request_dict:
                raise Exception(
                    "Question Answering Task requires 'question' and 'context' keys")
            args = (request_dict["question"], request_dict["context"])
            kwargs = query_kwargs

        elif self.task == Tasks.CONVERSATIONAL:
            conv = task_methods.create_conversation(request_dict, **query_kwargs)
            args = (conv, )
            kwargs = {}

        else:
            args = (request_dict['query'], )
            kwargs = query_kwargs

        return task_methods.run_inference(inference_pipeline, args, query_kwargs)

    def terminate(self):
        print(f"Terminating {self.deployment_name}...")
        del mii.non_persistent_models[self.deployment_name]


def terminate_restful_gateway(deployment_tag):
    deployments = _get_deployment_configs(deployment_tag)
    for deployment in deployments:
        mii_configs_dict = deployment[mii.constants.MII_CONFIGS_KEY]
        mii_configs = mii.config.MIIConfig(**mii_configs_dict)
        if mii_configs.enable_restful_api:
            requests.get(f"http://localhost:{mii_configs.restful_api_port}/terminate")
