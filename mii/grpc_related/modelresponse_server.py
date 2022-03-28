'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from concurrent import futures
import logging

import grpc
from .proto import modelresponse_pb2
from .proto import modelresponse_pb2_grpc
import sys


class ModelResponse(modelresponse_pb2_grpc.ModelResponseServicer):
    def __init__(self, inference_pipeline):
        self.inference_pipeline = inference_pipeline

    def GeneratorReply(self, request, context):
        response = self.inference_pipeline(request.request,
                                           do_sample=True,
                                           min_length=50)
        return modelresponse_pb2.SingleStringReply(response=f"{response}")

    def ClassificationReply(self, request, context):
        response = self.inference_pipeline(request.request, return_all_scores=True)
        return modelresponse_pb2.SingleStringReply(response=f"{response}")

    def QuestionAndAnswerReply(self, request, context):
        response = self.inference_pipeline(question=request.question,
                                           context=request.context)
        return modelresponse_pb2.SingleStringReply(response=f"{response}")


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
