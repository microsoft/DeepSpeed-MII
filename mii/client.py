# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import asyncio
import grpc
import requests
from typing import Dict, Any, Callable

from mii.config import get_mii_config, MIIConfig
from mii.constants import GRPC_MAX_MSG_SIZE, TaskType
from mii.grpc_related.proto import modelresponse_pb2, modelresponse_pb2_grpc
from mii.task_methods import TASK_METHODS_DICT


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
    def __init__(self, mii_config: MIIConfig, host: str = "localhost") -> None:
        self.mii_config = mii_config
        self.task = mii_config.model_config.task
        self.port = mii_config.port_number
        self.asyncio_loop = asyncio.get_event_loop()
        channel = create_channel(host, self.port)
        self.stub = modelresponse_pb2_grpc.ModelResponseStub(channel)

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    async def _request_async_response(self, request_dict, **query_kwargs):
        task_methods = TASK_METHODS_DICT[self.task]
        proto_request = task_methods.pack_request_to_proto(request_dict, **query_kwargs)
        proto_response = await getattr(self.stub, task_methods.method)(proto_request)
        return task_methods.unpack_response_from_proto(proto_response)

    async def _request_async_response_stream(self, request_dict, **query_kwargs):
        task_methods = TASK_METHODS_DICT[self.task]
        proto_request = task_methods.pack_request_to_proto(request_dict, **query_kwargs)
        assert hasattr(task_methods, "method_stream_out"), f"{self.task} does not support streaming response"
        async for response in getattr(self.stub,
                                      task_methods.method_stream_out)(proto_request):
            yield task_methods.unpack_response_from_proto(response)

    def generate(self,
                 prompt: str,
                 streaming_fn: Callable = None,
                 **query_kwargs: Dict[str,
                                      Any]):
        if not isinstance(prompt, str):
            raise RuntimeError(
                "MII client only supports a single query string, multi-string will be added soon"
            )
        request_dict = {"query": prompt}
        if streaming_fn is not None:
            return self._generate_stream(streaming_fn, request_dict, **query_kwargs)

        return self.asyncio_loop.run_until_complete(
            self._request_async_response(request_dict,
                                         **query_kwargs))

    def _generate_stream(self,
                         callback,
                         request_dict: Dict[str,
                                            str],
                         **query_kwargs: Dict[str,
                                              Any]):
        async def put_result():
            response_stream = self._request_async_response_stream(
                request_dict,
                **query_kwargs)

            while True:
                try:
                    response = await response_stream.__anext__()
                    callback(response)
                except StopAsyncIteration:
                    break

        self.asyncio_loop.run_until_complete(put_result())

    async def terminate_async(self):
        await self.stub.Terminate(
            modelresponse_pb2.google_dot_protobuf_dot_empty__pb2.Empty())

    def terminate_server(self):
        self.asyncio_loop.run_until_complete(self.terminate_async())
        if self.mii_config.enable_restful_api:
            requests.get(
                f"http://localhost:{self.mii_config.restful_api_port}/terminate")

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


def client(model_or_deployment_name: str) -> MIIClient:
    mii_config = get_mii_config(model_or_deployment_name)

    return MIIClient(mii_config)
