'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from concurrent import futures
import logging

import grpc
from .proto import modelresponse_pb2
from .proto import modelresponse_pb2_grpc
import sys
import time

from transformers import Conversation


class ModelResponse(modelresponse_pb2_grpc.ModelResponseServicer):
    def __init__(self, inference_pipeline):
        self.inference_pipeline = inference_pipeline

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

    def GeneratorReply(self, request, context):
        query_kwargs = self._unpack_proto_query_kwargs(request.query_kwargs)

        # unpack grpc list into py-list
        request = [r for r in request.request]

        start = time.time()
        batched_responses = self.inference_pipeline(request, **query_kwargs)
        end = time.time()

        # response is a list
        text_responses = []
        for response in batched_responses:
            text = response[0]['generated_text']
            text_responses.append(text)

        model_time = self._get_model_time(self.inference_pipeline.model, sum_times=True)

        val = modelresponse_pb2.MultiStringReply(response=text_responses,
                                                 time_taken=end - start,
                                                 model_time_taken=model_time)
        return val

    def ClassificationReply(self, request, context):
        query_kwargs = self._unpack_proto_query_kwargs(request.query_kwargs)
        start = time.time()
        response = self.inference_pipeline(request.request,
                                           return_all_scores=True,
                                           **query_kwargs)
        end = time.time()
        model_time = self._get_model_time(self.inference_pipeline.model)
        return modelresponse_pb2.SingleStringReply(response=f"{response}",
                                                   time_taken=end - start,
                                                   model_time_taken=model_time)

    def QuestionAndAnswerReply(self, request, context):
        query_kwargs = self._unpack_proto_query_kwargs(request.query_kwargs)
        start = time.time()
        response = self.inference_pipeline(question=request.question,
                                           context=request.context,
                                           **query_kwargs)
        end = time.time()
        model_time = self._get_model_time(self.inference_pipeline.model)
        return modelresponse_pb2.SingleStringReply(response=f"{response}",
                                                   time_taken=end - start,
                                                   model_time_taken=model_time)

    def FillMaskReply(self, request, context):
        query_kwargs = self._unpack_proto_query_kwargs(request.query_kwargs)
        start = time.time()
        response = self.inference_pipeline(request.request, **query_kwargs)
        end = time.time()

        model_time = self._get_model_time(self.inference_pipeline.model)
        return modelresponse_pb2.SingleStringReply(response=f"{response}",
                                                   time_taken=end - start,
                                                   model_time_taken=model_time)

    def TokenClassificationReply(self, request, context):
        query_kwargs = self._unpack_proto_query_kwargs(request.query_kwargs)
        start = time.time()
        response = self.inference_pipeline(request.request, **query_kwargs)
        end = time.time()
        model_time = self._get_model_time(self.inference_pipeline.model)
        return modelresponse_pb2.SingleStringReply(response=f"{response}",
                                                   time_taken=end - start,
                                                   model_time_taken=model_time)

    def ConversationalReply(self, request, context):
        query_kwargs = self._unpack_proto_query_kwargs(request.query_kwargs)
        start = time.time()
        conv = Conversation(text=request.text,
                            conversation_id=request.conversation_id,
                            past_user_inputs=request.past_user_inputs,
                            generated_responses=request.generated_responses,
                            **query_kwargs)
        self.inference_pipeline(conv)
        end = time.time()
        model_time = self._get_model_time(self.inference_pipeline.model)
        return modelresponse_pb2.ConversationReply(
            conversation_id=conv.uuid,
            past_user_inputs=conv.past_user_inputs,
            generated_responses=conv.generated_responses,
            time_taken=end - start,
            model_time_taken=model_time)


def serve(inference_pipeline, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
