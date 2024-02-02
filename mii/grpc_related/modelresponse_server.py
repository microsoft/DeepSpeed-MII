# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import asyncio
import queue
import sys
import threading
from concurrent import futures
from typing import Dict

import grpc
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

from mii.backend.client import create_channel
from mii.constants import (
    GenerationFinishReason,
    GRPC_MAX_MSG_SIZE,
    TERMINATE_METHOD,
    LB_MAX_WORKER_THREADS,
    SERVER_SHUTDOWN_TIMEOUT,
    STREAM_RESPONSE_QUEUE_TIMEOUT,
)
from mii.grpc_related.proto import modelresponse_pb2_grpc
from mii.grpc_related.task_methods import TASK_METHODS_DICT, TaskMethods


class ServiceBase(modelresponse_pb2_grpc.ModelResponseServicer):
    """
    Base class to provide common features of an inference server
    """
    def __init__(self):
        self._stop_event = threading.Event()

    def Terminate(self, request, context):
        self._stop_event.set()
        return google_dot_protobuf_dot_empty__pb2.Empty()

    def get_stop_event(self):
        return self._stop_event


class ModelResponse(ServiceBase):
    """
    Implementation class of an MII inference server
    """
    def __init__(self, async_pipeline=None):
        super().__init__()
        self.inference_pipeline = async_pipeline
        self.method_name_to_task = {m.method: t for t, m in TASK_METHODS_DICT.items()}
        self.lock = threading.Lock()

    def _get_task_methods(self, method_name: str) -> Dict[str, TaskMethods]:
        if method_name not in self.method_name_to_task:
            raise ValueError(f"unknown method: {method_name}")

        task = self.method_name_to_task[method_name]
        if task not in TASK_METHODS_DICT:
            raise ValueError(f"unknown task: {task}")

        task_methods = TASK_METHODS_DICT[task]
        return task_methods

    def GeneratorReply(self, request, context):
        task_methods = self._get_task_methods("GeneratorReply")

        prompts, kwargs = task_methods.unpack_request_from_proto(request)
        uids_put_order, uids_running, uids_complete_order, responses = [], [], [], []

        # Put requests for all prompts into the pipeline
        for p in prompts:
            request_kwargs = kwargs.copy()
            uid = self.inference_pipeline.put_request(p, request_kwargs)
            uids_put_order.append(uid)
            uids_running.append(uid)

        # Get responses from the pipeline as they are ready, flush finished uids
        # so new requests can be processed
        while uids_running:
            uid, response = self.inference_pipeline.get_response()
            # TODO: Ugly hack for multi-threading. Will be fixed when we refactor these methods
            if uid == -1:
                uid = uids_running[0]
            responses.append(response)
            self.inference_pipeline.flush_uid(uid)
            uids_complete_order.append(uids_put_order.index(uid))
            uids_running.remove(uid)

        # Sort responses in the order of prompts
        responses = [
            r for idx,
            r in sorted(zip(uids_complete_order,
                            responses),
                        key=lambda pair: pair[0])
        ]

        return task_methods.pack_response_to_proto(responses)

    def GeneratorReplyStream(self, request, context):
        task_methods = self._get_task_methods("GeneratorReply")

        prompts, kwargs = task_methods.unpack_request_from_proto(request)
        uid = self.inference_pipeline.put_request(prompts[0], kwargs)

        while True:
            response_uid, r = self.inference_pipeline.get_response()
            assert uid == response_uid, "uid mismatch"
            done = r.finish_reason != GenerationFinishReason.NONE
            response = task_methods.pack_response_to_proto([r])
            yield response
            if done:
                break

        self.inference_pipeline.flush_uid(uid)


class AtomicCounter:
    def __init__(self, initial_value=0):
        self.value = initial_value
        self.lock = threading.Lock()

    def get_and_increment(self):
        with self.lock:
            current_value = self.value
            self.value += 1
            return current_value

    def get(self):
        with self.lock:
            return self.value


def _get_grpc_method_name(method):
    return method.split("/")[-1]


class ParallelStubInvoker:
    """
    Invokes a gRPC method on multiple endpoints in parallel.
    This class aims to call gRPC methods without conversions between proto and python object.
    TensorParallelClient can be used for invocation with the conversions.
    """
    def __init__(self, host, ports):
        # Assumption: target services are all on the same host
        self.stubs = []
        for port in ports:
            channel = create_channel(host, port)
            stub = modelresponse_pb2_grpc.ModelResponseStub(channel)
            self.stubs.append(stub)

        self.asyncio_loop = asyncio.get_event_loop()

    async def _invoke_async(self, method_name, proto_request):
        responses = []
        if method_name == TERMINATE_METHOD:
            stubs = self.stubs
        else:
            stubs = [self.stubs[0]
                     ]  # Only the first stub is used for non-terminate methods
        for stub in stubs:
            method = getattr(stub, method_name)
            responses.append(method(proto_request))
        return await responses[0]

    def invoke(self, method_name, proto_request):
        # This is needed because gRPC calls from interceptor are launched from
        return asyncio.run_coroutine_threadsafe(
            self._invoke_async(method_name,
                               proto_request),
            self.asyncio_loop).result()

    def invoke_stream(self, method_name, proto_request, result_queue):
        async def invoke_stream_async():
            stub = self.stubs[0]  # Only the first stub is used for streaming
            method = getattr(stub, method_name)
            response = method(proto_request)

            async for r in response:
                result_queue.put(r)

        return asyncio.run_coroutine_threadsafe(invoke_stream_async(),
                                                self.asyncio_loop).result()


class LoadBalancingInterceptor(grpc.ServerInterceptor):
    def __init__(self, model_config):
        super().__init__()
        self.asyncio_loop = asyncio.get_event_loop()

        self.stubs = [
            ParallelStubInvoker(replica.hostname,
                                replica.tensor_parallel_ports)
            for replica in model_config.replica_configs
        ]
        self.counter = AtomicCounter()
        self.task = model_config.task

        # Start the asyncio loop in a separate thread
        def run_asyncio_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        threading.Thread(target=run_asyncio_loop, args=(self.asyncio_loop, )).start()

    def choose_stub(self, call_count):
        return self.stubs[call_count % len(self.stubs)]

    def intercept_service(self, continuation, handler_call_details):
        next_handler = continuation(handler_call_details)

        call_count = self.counter.get_and_increment()
        replica_index = call_count % len(self.stubs)

        def invoke_intercept_method(request_proto, context):
            method_name = _get_grpc_method_name(handler_call_details.method)

            if method_name == TERMINATE_METHOD:
                for stub in self.stubs:
                    stub.invoke(TERMINATE_METHOD,
                                google_dot_protobuf_dot_empty__pb2.Empty())
                self.asyncio_loop.call_soon_threadsafe(self.asyncio_loop.stop)
                return next_handler.unary_unary(request_proto, context)

            call_count = self.counter.get()
            replica_index = call_count % len(self.stubs)

            ret = self.stubs[replica_index].invoke(method_name, request_proto)
            return ret

        if next_handler.unary_unary is not None:
            return grpc.unary_unary_rpc_method_handler(
                invoke_intercept_method,
                request_deserializer=next_handler.request_deserializer,
                response_serializer=next_handler.response_serializer)
        else:
            method_name = _get_grpc_method_name(handler_call_details.method)
            result_queue = queue.Queue()

            def call_invoker(request_proto, context):
                self.stubs[replica_index].invoke_stream(method_name,
                                                        request_proto,
                                                        result_queue)

            def invoke_intercept_method_stream(request_proto, context):
                threading.Thread(target=call_invoker,
                                 args=(request_proto,
                                       context)).start()
                while True:
                    try:
                        response_proto = result_queue.get(
                            timeout=STREAM_RESPONSE_QUEUE_TIMEOUT)
                        yield response_proto
                        if response_proto.response[0].finish_reason != str(
                                GenerationFinishReason.NONE.value):
                            break
                    except queue.Empty:
                        print(
                            f"Haven't received a streaming response in {STREAM_RESPONSE_QUEUE_TIMEOUT} second(s)"
                        )
                        break

            return grpc.unary_stream_rpc_method_handler(
                invoke_intercept_method_stream,
                request_deserializer=next_handler.request_deserializer,
                response_serializer=next_handler.response_serializer)


def _do_serve(service_impl, port, interceptors=[]):
    stop_event = service_impl.get_stop_event()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=LB_MAX_WORKER_THREADS),
                         interceptors=interceptors,
                         options=[("grpc.max_send_message_length",
                                   GRPC_MAX_MSG_SIZE),
                                  ("grpc.max_receive_message_length",
                                   GRPC_MAX_MSG_SIZE)])
    modelresponse_pb2_grpc.add_ModelResponseServicer_to_server(service_impl, server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"About to start server")
    server.start()
    print(f"Started")
    stop_event.wait()
    server.stop(SERVER_SHUTDOWN_TIMEOUT)


def serve_inference(async_pipeline, port):
    async_pipeline.start()
    _do_serve(ModelResponse(async_pipeline=async_pipeline), port)
    async_pipeline.shutdown()


def serve_load_balancing(model_config, lb_port):
    _do_serve(ServiceBase(), lb_port, [LoadBalancingInterceptor(model_config)])


if __name__ == "__main__":
    import logging
    logging.basicConfig()
    serve_inference(None, sys.argv[1])
