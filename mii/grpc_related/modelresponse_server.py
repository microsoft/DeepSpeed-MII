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
from mii.constants import GRPC_MAX_MSG_SIZE


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

    def Txt2ImgReply(self, request, context):
        query_kwargs = self._unpack_proto_query_kwargs(request.query_kwargs)

        # unpack grpc list into py-list
        request = [r for r in request.request]

        start = time.time()
        response = self.inference_pipeline(request, **query_kwargs)
        end = time.time()

        images_bytes = []
        nsfw_content_detected = []
        response_count = len(response.images)
        for i in range(response_count):
            img = response.images[i]
            img_bytes = img.tobytes()
            images_bytes.append(img_bytes)
            nsfw_content_detected.append(response.nsfw_content_detected[i])
        img_mode = response.images[0].mode
        img_size_w, img_size_h = response.images[0].size

        val = modelresponse_pb2.ImageReply(images=images_bytes,
                                           nsfw_content_detected=nsfw_content_detected,
                                           mode=img_mode,
                                           size_w=img_size_w,
                                           size_h=img_size_h,
                                           time_taken=end - start)
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
