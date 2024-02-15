# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import asyncio
import grpc
import requests
import mii.legacy as mii
from .grpc_related.proto import legacymodelresponse_pb2 as modelresponse_pb2
from .grpc_related.proto import legacymodelresponse_pb2_grpc as modelresponse_pb2_grpc
from .constants import GRPC_MAX_MSG_SIZE, TaskType, DeploymentType, REQUIRED_KEYS_PER_TASK
from .method_table import GRPC_METHOD_TABLE
from .config import MIIConfig
from .utils import import_score_file


def _get_mii_config(deployment_name):
    mii_config = import_score_file(deployment_name, DeploymentType.LOCAL).mii_config
    return MIIConfig(**mii_config)


def mii_query_handle(deployment_name):
    """Get a query handle for a local deployment:

        mii/examples/local/gpt2-query-example.py
        mii/examples/local/roberta-qa-query-example.py

    Arguments:
        deployment_name: Name of the deployment. Used as an identifier for posting queries for ``LOCAL`` deployment.

    Returns:
        query_handle: A query handle with a single method `.query(request_dictionary)` using which queries can be sent to the model.
    """

    if deployment_name in mii.non_persistent_models:
        inference_pipeline, task = mii.non_persistent_models[deployment_name]
        return MIINonPersistentClient(task, deployment_name)

    mii_config = _get_mii_config(deployment_name)
    return MIIClient(mii_config.model_config.task,
                     "localhost", # TODO: This can probably be removed
                     mii_config.port_number)


def create_channel(host, port):
    return grpc.aio.insecure_channel(
        f"{host}:{port}",
        options=[
            ("grpc.max_send_message_length",
             GRPC_MAX_MSG_SIZE),
            ("grpc.max_receive_message_length",
             GRPC_MAX_MSG_SIZE),
        ],
    )


class MIIClient:
    """
    Client to send queries to a single endpoint.
    """
    def __init__(self, task, host, port):
        self.asyncio_loop = asyncio.get_event_loop()
        channel = create_channel(host, port)
        self.stub = modelresponse_pb2_grpc.ModelResponseStub(channel)
        self.task = task

    async def _request_async_response(self, request_dict, **query_kwargs):
        if self.task not in GRPC_METHOD_TABLE:
            raise ValueError(f"unknown task: {self.task}")

        task_methods = GRPC_METHOD_TABLE[self.task]
        proto_request = task_methods.pack_request_to_proto(request_dict, **query_kwargs)
        proto_response = await getattr(self.stub, task_methods.method)(proto_request)
        return task_methods.unpack_response_from_proto(proto_response)

    def query(self, request_dict, **query_kwargs):
        return self.asyncio_loop.run_until_complete(
            self._request_async_response(request_dict,
                                         **query_kwargs))

    async def terminate_async(self):
        await self.stub.Terminate(
            modelresponse_pb2.google_dot_protobuf_dot_empty__pb2.Empty())

    def terminate(self):
        self.asyncio_loop.run_until_complete(self.terminate_async())

    async def create_session_async(self, session_id):
        return await self.stub.CreateSession(
            modelresponse_pb2.SessionID(session_id=session_id))

    def create_session(self, session_id):
        assert (
            self.task == TaskType.TEXT_GENERATION
        ), f"Session creation only available for task '{TaskType.TEXT_GENERATION}'."
        return self.asyncio_loop.run_until_complete(
            self.create_session_async(session_id))

    async def destroy_session_async(self, session_id):
        await self.stub.DestroySession(modelresponse_pb2.SessionID(session_id=session_id)
                                       )

    def destroy_session(self, session_id):
        assert (
            self.task == TaskType.TEXT_GENERATION
        ), f"Session deletion only available for task '{TaskType.TEXT_GENERATION}'."
        self.asyncio_loop.run_until_complete(self.destroy_session_async(session_id))


class MIINonPersistentClient:
    def __init__(self, task, deployment_name):
        self.task = task
        self.deployment_name = deployment_name

    def query(self, request_dict, **query_kwargs):
        assert (
            self.deployment_name in mii.non_persistent_models
        ), f"deployment: {self.deployment_name} not found"
        task_methods = GRPC_METHOD_TABLE[self.task]
        inference_pipeline = mii.non_persistent_models[self.deployment_name][0]

        for key in REQUIRED_KEYS_PER_TASK[self.task]:
            assert key in request_dict, f"Task '{self.task}' requires '{key}' key"
        if self.task == TaskType.QUESTION_ANSWERING:
            args = (request_dict["question"], request_dict["context"])
            kwargs = query_kwargs
        elif self.task == TaskType.CONVERSATIONAL:
            conv = task_methods.create_conversation(request_dict)
            args = (conv, )
            kwargs = query_kwargs
        elif self.task == TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION:
            args = (request_dict["image"], request_dict["candidate_labels"])
            kwargs = query_kwargs
        elif self.task == TaskType.TEXT2IMG:
            args = (request_dict["prompt"], request_dict.get("negative_prompt", None))
            kwargs = query_kwargs
        elif self.task == TaskType.INPAINTING:
            negative_prompt = request_dict.get("negative_prompt", None)
            args = (request_dict["prompt"],
                    request_dict["image"],
                    request_dict["mask_image"],
                    negative_prompt)
            kwargs = query_kwargs
        else:
            args = (request_dict["query"], )
            kwargs = query_kwargs

        return task_methods.run_inference(inference_pipeline, args, query_kwargs)

    def terminate(self):
        print(f"Terminating {self.deployment_name}...")
        del mii.non_persistent_models[self.deployment_name]


def terminate_restful_gateway(deployment_name):
    mii_config = _get_mii_config(deployment_name)
    if mii_config.enable_restful_api:
        requests.get(f"http://localhost:{mii_config.restful_api_port}/terminate")
