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
from mii.deployment import allocate_processes, create_score_file, validate_deployment
from mii.config import DeploymentConfig


def _get_deployment_configs(deployment_tag):
    deployments = {}
    configs = mii.utils.import_score_file(deployment_tag).configs
    for deployment in configs.get(mii.constants.DEPLOYMENTS_KEY).values():
        deployment_name = deployment[mii.constants.DEPLOYMENT_NAME_KEY]
        deployments[deployment_name] = DeploymentConfig(**deployment)
    lb_config = configs.get(mii.constants.LOAD_BALANCER_CONFIG_KEY)
    model_path = configs.get(mii.constants.MODEL_PATH_KEY)
    port_map = configs.get(mii.constants.PORT_MAP_KEY)
    return deployments, lb_config, model_path, port_map


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

    deployments, lb_config, model_path, port_map = _get_deployment_configs(deployment_tag)
    mii_configs = None
    if len(deployments) > 0:
        mii_configs = getattr(next(iter(deployments.values())),
                              mii.constants.MII_CONFIGS_KEY)
    port_number = None if mii_configs == None else mii_configs.port_number
    if port_number:
        for deployment in deployments.values():
            assert getattr(deployment, mii.constants.MII_CONFIGS_KEY).port_number == port_number, f"All port numbers is each deployments mii_configs must match"

    return LBClient(deployments,
                    "localhost",
                    port_number,
                    lb_config,
                    model_path,
                    port_map,
                    deployment_tag)


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
        self.mr_stub = None
        self.channel = None
        self.host = host
        if port is not None:
            self.channel = create_channel(host, port)
            self.mr_stub = modelresponse_pb2_grpc.ModelResponseStub(self.channel)
        self.deployments = deployments

    def _get_deployment_task(self, deployment_name=None):
        task = None
        if deployment_name is None or deployment_name == mii.constants.MII_TERMINATE_DEP_KEY:  #mii.terminate() or single model
            assert len(self.deployments) == 1, "Must pass deployment_name to query when using multiple deployments"
            deployment = next(iter(self.deployments.values()))
            deployment_name = getattr(deployment, mii.constants.DEPLOYMENT_NAME_KEY)
            task = getattr(deployment, mii.constants.TASK_NAME_KEY)
        else:
            if deployment_name in self.deployments:
                deployment = self.deployments[deployment_name]
                task = getattr(deployment, mii.constants.TASK_NAME_KEY)
            else:
                assert False, f"{deployment_name} not found in list of deployments"
        return deployment_name, task

    async def _request_async_response(self, request_dict, task, **query_kwargs):
        if task not in GRPC_METHOD_TABLE:
            raise ValueError(f"unknown task: {task}")

        task_methods = GRPC_METHOD_TABLE[task]
        proto_request = task_methods.pack_request_to_proto(request_dict, **query_kwargs)
        proto_response = await getattr(self.mr_stub, task_methods.method)(proto_request)
        return task_methods.unpack_response_from_proto(proto_response)

    def query(self, request_dict, **query_kwargs):
        deployment_name = request_dict.get(mii.constants.DEPLOYMENT_NAME_KEY)
        deployment_name, task = self._get_deployment_task(deployment_name)
        request_dict['deployment_name'] = deployment_name
        return self.asyncio_loop.run_until_complete(
            self._request_async_response(request_dict,
                                         task,
                                         **query_kwargs))

    async def terminate_async(self):
        await self.lb_stub.Terminate(
            modelresponse_pb2.google_dot_protobuf_dot_empty__pb2.Empty())

    def terminate(self):
        self.asyncio_loop.run_until_complete(self.terminate_async())

    async def create_session_async(self, session_id):
        return await self.mr_stub.CreateSession(
            modelresponse_pb2.SessionID(session_id=session_id))

    def create_session(self, session_id, deployment_name=None):
        if len(self.deployments > 1):
            assert deployment_name is not None, "Deployment name must be passed in to create session when there are multiple models"
        deployment_name, task = self._get_deployment_task(deployment_name)
        assert task == Tasks.TEXT_GENERATION, f"Session creation only available for task '{Tasks.TEXT_GENERATION}'."
        return self.asyncio_loop.run_until_complete(
            self.create_session_async(session_id))

    async def destroy_session_async(self, session_id):
        await self.mr_stub.DestroySession(
            modelresponse_pb2.SessionID(session_id=session_id))

    def destroy_session(self, session_id, deployment_name=None):
        if len(self.deployments > 1):
            assert deployment_name is not None, "Deployment name must be passed in to destroy session when there are multiple models"
        deployment_name, task = self._get_deployment_task(deployment_name)
        assert task == Tasks.TEXT_GENERATION, f"Session deletion only available for task '{Tasks.TEXT_GENERATION}'."
        self.asyncio_loop.run_until_complete(self.destroy_session_async(session_id))


class LBClient(MIIClient):
    def __init__(self,
                 deployments,
                 host,
                 port,
                 lb_config=None,
                 model_path=None,
                 port_map=None,
                 deployment_tag=None):
        super().__init__(deployments, host, port)
        self.lb_stub = None
        if port is not None:
            channel = create_channel(host, port) if not self.channel else self.channel
            self.lb_stub = modelresponse_pb2_grpc.DeploymentManagementStub(channel)
        self.lb_config = lb_config
        self.model_path = model_path
        self.port_map = port_map if port_map is not None else {}
        self.deployment_tag = deployment_tag

    async def add_models_async(self, proto_request):
        await getattr(self.lb_stub, "AddDeployment")(proto_request)

    def add_models(self,
                   task=None,
                   model=None,
                   deployment_name=None,
                   enable_deepspeed=True,
                   enable_zero=False,
                   ds_config=None,
                   mii_config={},
                   deployments=[],
                   deployment_type=DeploymentType.LOCAL,
                   model_path=None,
                   version=1):

        _, deployments = validate_deployment(task=task,
                                             model=model,
                                             deployment_name=deployment_name,
                                             enable_deepspeed=enable_deepspeed,
                                             enable_zero=enable_zero,
                                             ds_config=ds_config,
                                             mii_config=mii_config,
                                             deployment_tag=self.deployment_tag,
                                             deployments=deployments,
                                             deployment_type=deployment_type,
                                             model_path=model_path,
                                             version=version)

        if not deployments:  #Empty deployment
            return None

        deps = {
            getattr(deployment,
                    mii.constants.DEPLOYMENT_NAME_KEY): deployment
            for deployment in deployments
        }
        lb_config, self.port_map = allocate_processes(deps, self.port_map)
        lb_enabled = True if len(self.deployments) else False
        if self.lb_config is not None:
            self.lb_config.replica_configs.extend(lb_config.replica_configs)
        else:
            self.lb_config = lb_config
        for deployment in deployments:
            self.deployments[getattr(deployment,
                                     mii.constants.DEPLOYMENT_NAME_KEY)] = deployment
        if self.model_path is None and deployment_type == DeploymentType.LOCAL:
            self.model_path = mii.constants.MII_MODEL_PATH_DEFAULT
        elif self.model_path is None and deployment_type == DeploymentType.AML:
            model_path = "model"
        create_score_file(deployment_tag=self.deployment_tag,
                          deployment_type=deployment_type,
                          deployments=deps,
                          model_path=self.model_path,
                          port_map=self.port_map,
                          lb_config=lb_config,
                          deployed=lb_enabled)
        if deployment_type == DeploymentType.LOCAL:
            mii.utils.import_score_file(self.deployment_tag).init()
        if self.lb_stub is None:
            self.port_number = getattr(next(iter(self.deployments.values())),
                                       mii.constants.MII_CONFIGS_KEY).port_number
            self.channel = create_channel(self.host, self.port_number)
            self.lb_stub = modelresponse_pb2_grpc.DeploymentManagementStub(self.channel)
            if not self.mr_stub:
                self.mr_stub = modelresponse_pb2_grpc.ModelResponseStub(self.channel)
        for replica in lb_config.replica_configs:
            request_proto = modelresponse_pb2.AddDeployRequest(
                task=replica.task,
                deployment_name=replica.deployment_name,
                hostname=replica.hostname,
                tensor_parallel_ports=replica.tensor_parallel_ports,
                torch_dist_port=replica.torch_dist_port,
                gpu_indices=replica.gpu_indices)

            self.asyncio_loop.run_until_complete(self.add_models_async(request_proto))

    async def delete_model_async(self, proto_request):
        await getattr(self.lb_stub, "DeleteDeployment")(proto_request)

    def delete_model(self, deployment_name):
        if deployment_name in self.deployments:
            request_proto = modelresponse_pb2.DeleteDeployRequest(
                deployment_name=deployment_name)
            self.asyncio_loop.run_until_complete(self.delete_model_async(request_proto))
            del self.deployments[deployment_name]
            return None
        assert False, f"Deployment: {deployment_name} not found"


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
    deployments, _, _, _ = _get_deployment_configs(deployment_tag)
    for deployment in deployments.values():
        mii_configs = getattr(deployment, mii.constants.MII_CONFIGS_KEY)
        if mii_configs.enable_restful_api:
            requests.get(f"http://localhost:{mii_configs.restful_api_port}/terminate")
