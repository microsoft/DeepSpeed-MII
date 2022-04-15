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

    def GeneratorReply(self, request, context):
        start = time.time()
        response = self.inference_pipeline(request.request,
                                           do_sample=True,
                                           min_length=50)
        end = time.time()
        return modelresponse_pb2.SingleStringReply(response=f"{response}")

    def ClassificationReply(self, request, context):
        response = self.inference_pipeline(request.request, return_all_scores=True)
        return modelresponse_pb2.SingleStringReply(response=f"{response}")

    def QuestionAndAnswerReply(self, request, context):
        response = self.inference_pipeline(question=request.question,
                                           context=request.context)
        return modelresponse_pb2.SingleStringReply(response=f"{response}")

    def FillMaskReply(self, request, context):
        response = self.inference_pipeline(request.request)
        return modelresponse_pb2.SingleStringReply(response=f"{response}")

    def TokenClassificationReply(self, request, context):
        response = self.inference_pipeline(request.request)
        return modelresponse_pb2.SingleStringReply(response=f"{response}")

    def ConversationalReply(self, request, context):
        conv = Conversation(text=request.text,
                            conversation_id=request.conversation_id,
                            past_user_inputs=request.past_user_inputs,
                            generated_responses=request.generated_responses)
        self.inference_pipeline(conv)
        return modelresponse_pb2.ConversationReply(
            conversation_id=conv.uuid,
            past_user_inputs=conv.past_user_inputs,
            generated_responses=conv.generated_responses)


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
