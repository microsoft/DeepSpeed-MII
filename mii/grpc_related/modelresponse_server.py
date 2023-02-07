'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import asyncio
from concurrent import futures
import logging

import grpc
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from .proto import modelresponse_pb2_grpc
import sys
import threading
import time

from mii.constants import GRPC_MAX_MSG_SIZE, TERMINATE_METHOD, LB_MAX_WORKER_THREADS, SERVER_SHUTDOWN_TIMEOUT
from mii.method_table import GRPC_METHOD_TABLE
from mii.client import create_channel
from mii.utils import get_task


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
    def __init__(self, inference_pipeline):
        super().__init__()
        self.inference_pipeline = inference_pipeline
        self.method_name_to_task = {m["method"]: t for t, m in GRPC_METHOD_TABLE.items()}

    def _get_model_time(self, model, sum_times=False):
        model_times = []
        # Only grab model times if profiling was enabled/exists
        if getattr(model, "model_profile_enabled", False):
            model_times = model.model_times()

        if len(model_times) > 0:
            if sum_times:
                model_time = sum(model_times)
            else:
                # Unclear how to combine values, so just grab the most recent one
                model_time = model_times[-1]
        else:
            # no model times were captured
            model_time = -1
        return model_time

    def _run_inference(self, method_name, request_proto):
        if method_name not in self.method_name_to_task:
            raise ValueError(f"unknown method: {method_name}")

        task = self.method_name_to_task[method_name]
        if task not in GRPC_METHOD_TABLE:
            raise ValueError(f"unknown task: {task}")

        conversions = GRPC_METHOD_TABLE[task]
        args, kwargs = conversions["unpack_request_from_proto"](request_proto)

        start = time.time()
        response = self.inference_pipeline(*args, **kwargs)
        end = time.time()

        model_time = self._get_model_time(self.inference_pipeline.model,
                                          sum_times=True) if hasattr(
                                              self.inference_pipeline,
                                              "model") else -1

        return conversions["pack_response_to_proto"](response, end - start, model_time)

    def GeneratorReply(self, request, context):
        return self._run_inference("GeneratorReply", request)

    def Txt2ImgReply(self, request, context):
        return self._run_inference("Txt2ImgReply", request)

    def ClassificationReply(self, request, context):
        return self._run_inference("ClassificationReply", request)

    def QuestionAndAnswerReply(self, request, context):
        return self._run_inference("QuestionAndAnswerReply", request)

    def FillMaskReply(self, request, context):
        return self._run_inference("FillMaskReply", request)

    def TokenClassificationReply(self, request, context):
        return self._run_inference("TokenClassificationReply", request)

    def ConversationalReply(self, request, context):
        return self._run_inference("ConversationalReply", request)


class AtomicCounter:
    def __init__(self, initial_value=0):
        self.value = initial_value
        self.lock = threading.Lock()

    def get_and_increment(self):
        with self.lock:
            current_value = self.value
            self.value += 1
            return current_value


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
        for stub in self.stubs:
            method = getattr(stub, method_name)
            responses.append(method(proto_request))
        return await responses[0]

    def invoke(self, method_name, proto_request):
        # This is needed because gRPC calls from interceptor are launched from
        return asyncio.run_coroutine_threadsafe(
            self._invoke_async(method_name,
                               proto_request),
            self.asyncio_loop).result()


class LoadBalancingInterceptor(grpc.ServerInterceptor):
    def __init__(self, task_name, replica_deployment):
        super().__init__()
        self.asyncio_loop = asyncio.get_event_loop()

        self.stubs = [
            ParallelStubInvoker(replica[0],
                                replica[1]) for replica in replica_deployment
        ]
        self.counter = AtomicCounter()
        self.task = get_task(task_name)

        # Start the asyncio loop in a separate thread
        def run_asyncio_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        threading.Thread(target=run_asyncio_loop, args=(self.asyncio_loop, )).start()

    def choose_stub(self, call_count):
        return self.stubs[call_count % len(self.stubs)]

    def intercept_service(self, continuation, handler_call_details):
        next_handler = continuation(handler_call_details)
        assert next_handler.unary_unary is not None

        def invoke_intercept_method(request_proto, context):
            method_name = _get_grpc_method_name(handler_call_details.method)

            if method_name == TERMINATE_METHOD:
                for stub in self.stubs:
                    stub.invoke(TERMINATE_METHOD,
                                google_dot_protobuf_dot_empty__pb2.Empty())
                self.asyncio_loop.call_soon_threadsafe(self.asyncio_loop.stop)
                return next_handler.unary_unary(request_proto, context)

            call_count = self.counter.get_and_increment()
            ret = self.stubs[call_count % len(self.stubs)].invoke(
                method_name,
                request_proto)
            return ret

        return grpc.unary_unary_rpc_method_handler(
            invoke_intercept_method,
            request_deserializer=next_handler.request_deserializer,
            response_serializer=next_handler.response_serializer)


def _do_serve(service_impl, port, interceptors=[]):
    stop_event = service_impl.get_stop_event()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=LB_MAX_WORKER_THREADS),
                         interceptors=interceptors,
                         options=[('grpc.max_send_message_length',
                                   GRPC_MAX_MSG_SIZE),
                                  ('grpc.max_receive_message_length',
                                   GRPC_MAX_MSG_SIZE)])
    modelresponse_pb2_grpc.add_ModelResponseServicer_to_server(service_impl, server)
    server.add_insecure_port(f'[::]:{port}')
    print(f"About to start server")
    server.start()
    print(f"Started")
    stop_event.wait()
    server.stop(SERVER_SHUTDOWN_TIMEOUT)


def serve_inference(inference_pipeline, port):
    _do_serve(ModelResponse(inference_pipeline), port)


def serve_load_balancing(task_name, mii_configs):
    _do_serve(ServiceBase(),
              mii_configs.port_number,
              [LoadBalancingInterceptor(task_name,
                                        mii_configs.replica_deployment)])


if __name__ == '__main__':
    logging.basicConfig()
    print(sys.argv[1])
    serve_inference(None, sys.argv[1])
