# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from transformers import Conversation
from abc import ABC, abstractmethod

from mii.constants import Tasks
from mii.grpc_related.proto import modelresponse_pb2
from mii.utils import kwarg_dict_to_proto, unpack_proto_query_kwargs
from mii.models.utils import ImageResponse


def single_string_request_to_proto(self, request_dict, **query_kwargs):
    return modelresponse_pb2.SingleStringRequest(
        request=request_dict['query'],
        query_kwargs=kwarg_dict_to_proto(query_kwargs))


def single_string_response_to_proto(self, response, time_taken, model_time_taken):
    return modelresponse_pb2.SingleStringReply(response=f"{response}",
                                               time_taken=time_taken,
                                               model_time_taken=model_time_taken)


def multi_string_request_to_proto(self, request_dict, **query_kwargs):
    return modelresponse_pb2.MultiStringRequest(
        request=request_dict['query'] if isinstance(request_dict['query'],
                                                    list) else [request_dict['query']],
        query_kwargs=kwarg_dict_to_proto(query_kwargs))


def proto_request_to_single_input(self, request):
    args = (request.request, )
    kwargs = unpack_proto_query_kwargs(request.query_kwargs)
    return args, kwargs


def proto_request_to_list(self, request):
    args = ([r for r in request.request], )
    kwargs = unpack_proto_query_kwargs(request.query_kwargs)
    return args, kwargs


class TaskMethods(ABC):
    @property
    @abstractmethod
    def method(self):
        ...

    def pack_request_to_proto(self, request_dict, **query_kwargs):
        return request_dict, query_kwargs

    def unpack_request_from_proto(self, request):
        return request

    def run_inference(self, inference_pipeline, args, kwargs, is_non_persistent=False):
        if is_non_persistent:
            return inference_pipeline(args, **kwargs)
        else:
            return inference_pipeline(*args, **kwargs)

    def pack_response_to_proto(self, response, time_taken, model_time_taken):
        return response, time_taken, model_time_taken

    def unpack_response_from_proto(self, response):
        return response


class TextGenerationMethods(TaskMethods):
    session_context = {}

    @property
    def method(self):
        return "GeneratorReply"

    pack_request_to_proto = multi_string_request_to_proto
    unpack_request_from_proto = proto_request_to_list

    def create_session(self, session_id):
        if session_id in self.session_context:
            raise ValueError(f"session {session_id} already exists")
        self.session_context[session_id] = None

    def destroy_session(self, session_id):
        if session_id not in self.session_context:
            raise ValueError(f"session {session_id} does not exist")
        del self.session_context[session_id]

    def preprocess_session(self, session_id, args):
        if session_id not in self.session_context:
            raise ValueError(f"session {session_id} does not exist")
        if self.session_context[session_id] is None:
            self.session_context[session_id] = ""
        if len(args[0]) != 1:
            raise ValueError(f"You can pass only one prompt with a session_id")

        args = ([self.session_context[session_id] + args[0][0]], )
        return args

    def run_inference(self, inference_pipeline, args, kwargs, is_non_persistent=False):
        session_id = kwargs.pop("session_id", None)
        if session_id:
            args = self.preprocess_session(session_id, args)
        response = inference_pipeline(
            *args,
            **kwargs) if not is_non_persistent else inference_pipeline(args,
                                                                       **kwargs)

        if session_id:
            response = self.postprocess_session(session_id, args, response)

        return response

    def postprocess_session(self, session_id, args, response):
        generated_text = response[0][0]["generated_text"]
        self.session_context[session_id] = generated_text
        response[0][0]["generated_text"] = generated_text[len(args[0][0]):]
        return response

    def pack_response_to_proto(self, response, time_taken, model_time_taken):
        text_responses = []
        for response in response:
            text = response[0]['generated_text']
            text_responses.append(text)

        return modelresponse_pb2.MultiStringReply(response=text_responses,
                                                  time_taken=time_taken,
                                                  model_time_taken=model_time_taken)


class TextClassificationMethods(TaskMethods):
    @property
    def method(self):
        return "ClassificationReply"

    pack_request_to_proto = single_string_request_to_proto
    unpack_request_from_proto = proto_request_to_single_input
    pack_response_to_proto = single_string_response_to_proto


class QuestionAnsweringMethods(TaskMethods):
    @property
    def method(self):
        return "QuestionAndAnswerReply"

    pack_response_to_proto = single_string_response_to_proto

    def pack_request_to_proto(self, request_dict, **query_kwargs):
        return modelresponse_pb2.QARequest(
            question=request_dict['question'],
            context=request_dict['context'],
            query_kwargs=kwarg_dict_to_proto(query_kwargs))

    def unpack_request_from_proto(self, request):
        kwargs = unpack_proto_query_kwargs(request.query_kwargs)
        kwargs["question"] = request.question
        kwargs["context"] = request.context
        args = ()
        return args, kwargs


class FillMaskMethods(TaskMethods):
    @property
    def method(self):
        return "FillMaskReply"

    pack_request_to_proto = single_string_request_to_proto
    unpack_request_from_proto = proto_request_to_single_input
    pack_response_to_proto = single_string_response_to_proto

    def run_inference(self, inference_pipeline, args, kwargs):
        return inference_pipeline(args, **kwargs)


class TokenClassificationMethods(TaskMethods):
    @property
    def method(self):
        return "TokenClassificationReply"

    pack_request_to_proto = single_string_request_to_proto
    unpack_request_from_proto = proto_request_to_single_input
    pack_response_to_proto = single_string_response_to_proto


class ConversationalMethods(TaskMethods):
    @property
    def method(self):
        return "ConversationalReply"

    def pack_response_to_proto(self, conv, time_taken, model_time_taken):
        return modelresponse_pb2.ConversationReply(
            conversation_id=conv.uuid,
            past_user_inputs=conv.past_user_inputs,
            generated_responses=conv.generated_responses,
            time_taken=time_taken,
            model_time_taken=model_time_taken)

    def unpack_request_from_proto(self, request):
        kwargs = unpack_proto_query_kwargs(request.query_kwargs)
        conv = Conversation(text=request.text,
                            conversation_id=request.conversation_id,
                            past_user_inputs=request.past_user_inputs,
                            generated_responses=request.generated_responses,
                            **kwargs)
        args = (conv, )
        kwargs = {}
        return args, kwargs

    def pack_request_to_proto(self, request_dict, **query_kwargs):
        return modelresponse_pb2.ConversationRequest(
            text=request_dict['text'],
            conversation_id=request_dict['conversation_id']
            if 'conversation_id' in request_dict else None,
            past_user_inputs=request_dict['past_user_inputs'],
            generated_responses=request_dict['generated_responses'],
            query_kwargs=kwarg_dict_to_proto(query_kwargs))


class Text2ImgMethods(TaskMethods):
    @property
    def method(self):
        return "Txt2ImgReply"

    pack_request_to_proto = multi_string_request_to_proto
    unpack_request_from_proto = proto_request_to_list

    def pack_response_to_proto(self, response, time_taken, model_time_taken):
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

        return modelresponse_pb2.ImageReply(images=images_bytes,
                                            nsfw_content_detected=nsfw_content_detected,
                                            mode=img_mode,
                                            size_w=img_size_w,
                                            size_h=img_size_h,
                                            time_taken=time_taken)

    def unpack_response_from_proto(self, response):
        return ImageResponse(response)


GRPC_METHOD_TABLE = {
    Tasks.TEXT_GENERATION: TextGenerationMethods(),
    Tasks.TEXT_CLASSIFICATION: TextClassificationMethods(),
    Tasks.QUESTION_ANSWERING: QuestionAnsweringMethods(),
    Tasks.FILL_MASK: FillMaskMethods(),
    Tasks.TOKEN_CLASSIFICATION: TokenClassificationMethods(),
    Tasks.CONVERSATIONAL: ConversationalMethods(),
    Tasks.TEXT2IMG: Text2ImgMethods(),
}
