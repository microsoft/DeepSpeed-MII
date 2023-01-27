'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from concurrent import futures
import logging

import grpc
from .proto import modelresponse_pb2_grpc
import sys
import time

from mii.constants import GRPC_MAX_MSG_SIZE
from mii.method_table import GRPC_METHOD_TABLE


class ModelResponse(modelresponse_pb2_grpc.ModelResponseServicer):
    def __init__(self, inference_pipeline):
        self.inference_pipeline = inference_pipeline
        self.method_name_to_task = {m["method"]: t for t, m in GRPC_METHOD_TABLE.items()}

    def _unpack_proto_query_kwargs(self, query_kwargs):
        query_kwargs = {
            k: getattr(v,
                       v.WhichOneof("oneof_values"))
            for k,
            v in query_kwargs.items()
        }
        return query_kwargs

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


def serve(inference_pipeline, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length',
                                   GRPC_MAX_MSG_SIZE),
                                  ('grpc.max_receive_message_length',
                                   GRPC_MAX_MSG_SIZE)])
    modelresponse_pb2_grpc.add_ModelResponseServicer_to_server(
        ModelResponse(inference_pipeline),
        server)
    server.add_insecure_port(f'[::]:{port}')
    print(f"About to start server")
    server.start()
    print(f"Started")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    print(sys.argv[1])
    serve(None, sys.argv[1])
