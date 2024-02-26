# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import asyncio
import grpc
import requests
from typing import Dict, Any, Callable, List, Union

from mii.batching.data_classes import Response
from mii.config import MIIConfig
from mii.constants import GRPC_MAX_MSG_SIZE
from mii.grpc_related.proto import modelresponse_pb2, modelresponse_pb2_grpc
from mii.grpc_related.task_methods import TASK_METHODS_DICT


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
    Client for sending generation requests to a persistent deployment created
    with :func:`mii.serve`. Use :func:`mii.client` to create an instance of this
    class.

    :param mii_config: MII config for the persistent deployment to connect with.
    :param host: hostname where the persistent deployment is running.
    """
    def __init__(self, mii_config: MIIConfig, host: str = "localhost") -> None:
        self.mii_config = mii_config
        self.task = mii_config.model_config.task
        self.port = mii_config.port_number
        self.asyncio_loop = asyncio.get_event_loop()
        channel = create_channel(host, self.port)
        self.stub = modelresponse_pb2_grpc.ModelResponseStub(channel)

    def __call__(self, *args, **kwargs) -> List[Response]:
        """
        All args and kwargs get passed directly to
        :meth:`~mii.backend.client.MIIClient.generate`.

        :return: A list of :class:`Response` objects containing the generated
            text for all prompts.
        """
        return self.generate(*args, **kwargs)

    async def _request_async_response(self, prompts, **query_kwargs):
        task_methods = TASK_METHODS_DICT[self.task]
        proto_request = task_methods.pack_request_to_proto(prompts, **query_kwargs)
        proto_response = await getattr(self.stub, task_methods.method)(proto_request)
        return task_methods.unpack_response_from_proto(proto_response)

    async def _request_async_response_stream(self, prompts, **query_kwargs):
        task_methods = TASK_METHODS_DICT[self.task]
        proto_request = task_methods.pack_request_to_proto(prompts, **query_kwargs)
        assert hasattr(task_methods, "method_stream_out"), f"{self.task} does not support streaming response"
        async for response in getattr(self.stub,
                                      task_methods.method_stream_out)(proto_request):
            yield task_methods.unpack_response_from_proto(response)

    def generate(self,
                 prompts: Union[str,
                                List[str]],
                 streaming_fn: Callable = None,
                 **generate_kwargs: Dict) -> List[Response]:
        """
        Generates text for the given prompts.

        :param prompts: The string or list of strings used as prompts for generation.
        :param streaming_fn: Streaming support is currently a WIP.
        :param \**generate_kwargs: Generation keywords. A full list can be found here.

        :return: A list of :class:`Response` objects containing the generated
            text for all prompts.
        """ # noqa: W605
        if isinstance(prompts, str):
            prompts = [prompts]
        if streaming_fn is not None:
            if len(prompts) > 1:
                raise RuntimeError(
                    "MII client streaming only supports a single prompt input.")
            generate_kwargs["stream"] = True
            return self._generate_stream(streaming_fn, prompts, **generate_kwargs)

        return self.asyncio_loop.run_until_complete(
            self._request_async_response(prompts,
                                         **generate_kwargs))

    def _generate_stream(self,
                         callback,
                         prompts: List[str],
                         **query_kwargs: Dict[str,
                                              Any]) -> None:
        async def put_result():
            response_stream = self._request_async_response_stream(
                prompts,
                **query_kwargs)

            while True:
                try:
                    response = await response_stream.__anext__()
                    callback(response)
                except StopAsyncIteration:
                    break

        self.asyncio_loop.run_until_complete(put_result())

    async def terminate_async(self) -> None:
        await self.stub.Terminate(
            modelresponse_pb2.google_dot_protobuf_dot_empty__pb2.Empty())

    def terminate_server(self) -> None:
        """
        Terminates the persistent deployment server. This can be called from any
        client.
        """
        self.asyncio_loop.run_until_complete(self.terminate_async())
        if self.mii_config.enable_restful_api:
            requests.get(
                f"http://localhost:{self.mii_config.restful_api_port}/terminate")
