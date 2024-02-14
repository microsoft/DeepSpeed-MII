# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import uuid

from abc import ABC, abstractmethod
from transformers import Conversation
from mii.legacy.constants import TaskType
from mii.legacy.grpc_related.proto import legacymodelresponse_pb2 as modelresponse_pb2
from mii.legacy.utils import kwarg_dict_to_proto, unpack_proto_query_kwargs
from mii.legacy.models.utils import ImageResponse, convert_bytes_to_pil_image


def single_string_request_to_proto(self, request_dict, **query_kwargs):
    return modelresponse_pb2.SingleStringRequest(
        request=request_dict["query"],
        query_kwargs=kwarg_dict_to_proto(query_kwargs))


def single_string_response_to_proto(self, response, time_taken, model_time_taken):
    return modelresponse_pb2.SingleStringReply(response=f"{response}",
                                               time_taken=time_taken,
                                               model_time_taken=model_time_taken)


def multi_string_request_to_proto(self, request_dict, **query_kwargs):
    return modelresponse_pb2.MultiStringRequest(
        request=request_dict["query"] if isinstance(request_dict["query"],
                                                    list) else [request_dict["query"]],
        query_kwargs=kwarg_dict_to_proto(query_kwargs),
    )


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

    def run_inference(self, inference_pipeline, args, kwargs):
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

    def run_inference(self, inference_pipeline, args, kwargs):
        session_id = kwargs.pop("session_id", None)
        if session_id:
            args = self.preprocess_session(session_id, args)
        response = inference_pipeline(*args, **kwargs)

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
            text = response[0]["generated_text"]
            text_responses.append(text)

        return modelresponse_pb2.MultiStringReply(
            response=text_responses,
            time_taken=time_taken,
            model_time_taken=model_time_taken,
        )


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
            question=request_dict["question"],
            context=request_dict["context"],
            query_kwargs=kwarg_dict_to_proto(query_kwargs),
        )

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

    def create_conversation(self, request):
        if isinstance(request, dict):
            assert 'text' in request and 'past_user_inputs' in request and 'generated_responses' in request, "Conversation requires 'text', 'past_user_inputs', and 'generated_responses' keys"
            text = request['text']
            conversation_id = request[
                'conversation_id'] if 'conversation_id' in request else ""
            past_user_inputs = request['past_user_inputs']
            generated_responses = request['generated_responses']

        else:
            text = getattr(request, 'text')
            conversation_id = getattr(request, 'conversation_id')
            past_user_inputs = getattr(request, 'past_user_inputs')
            generated_responses = getattr(request, 'generated_responses')

        # Create UUID from conversation ID
        conversation_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(conversation_id))

        conv = Conversation(text=text,
                            conversation_id=conversation_id,
                            past_user_inputs=past_user_inputs,
                            generated_responses=generated_responses)
        return conv

    def pack_response_to_proto(self, conv, time_taken, model_time_taken):
        return modelresponse_pb2.ConversationReply(
            conversation_id=str(conv.uuid),
            past_user_inputs=conv.past_user_inputs,
            generated_responses=conv.generated_responses,
            time_taken=time_taken,
            model_time_taken=model_time_taken,
        )

    def unpack_request_from_proto(self, request):
        kwargs = unpack_proto_query_kwargs(request.query_kwargs)
        conv = self.create_conversation(request)
        args = (conv, )
        return args, kwargs

    def pack_request_to_proto(self, request_dict, **query_kwargs):
        return modelresponse_pb2.ConversationRequest(
            text=request_dict['text'],
            conversation_id=str(request_dict['conversation_id'])
            if 'conversation_id' in request_dict else "",
            past_user_inputs=request_dict['past_user_inputs'],
            generated_responses=request_dict['generated_responses'],
            query_kwargs=kwarg_dict_to_proto(query_kwargs))


class Text2ImgMethods(TaskMethods):
    @property
    def method(self):
        return "Txt2ImgReply"

    def run_inference(self, inference_pipeline, args, kwargs):
        prompt, negative_prompt = args
        return inference_pipeline(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  **kwargs)

    def pack_request_to_proto(self, request_dict, **query_kwargs):
        prompt = request_dict["prompt"]
        prompt = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt = request_dict.get("negative_prompt", [""] * len(prompt))
        negative_prompt = [negative_prompt] if isinstance(negative_prompt,
                                                          str) else negative_prompt

        return modelresponse_pb2.Text2ImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            query_kwargs=kwarg_dict_to_proto(query_kwargs),
        )

    def pack_response_to_proto(self, response, time_taken, model_time_taken):
        images_bytes = []
        nsfw_content_detected = []
        response_count = len(response.images)
        for i in range(response_count):
            img = response.images[i]
            img_bytes = img.tobytes()
            images_bytes.append(img_bytes)
            nsfw_content_detected.append(response.nsfw_content_detected[i] if response.
                                         nsfw_content_detected else False)
        img_mode = response.images[0].mode
        img_size_w, img_size_h = response.images[0].size

        return modelresponse_pb2.ImageReply(
            images=images_bytes,
            nsfw_content_detected=nsfw_content_detected,
            mode=img_mode,
            size_w=img_size_w,
            size_h=img_size_h,
            time_taken=time_taken,
        )

    def unpack_response_from_proto(self, response):
        return ImageResponse(response)

    def unpack_request_from_proto(self, request):
        kwargs = unpack_proto_query_kwargs(request.query_kwargs)
        args = (list(request.prompt), list(request.negative_prompt))
        return args, kwargs


class ZeroShotImgClassificationMethods(TaskMethods):
    @property
    def method(self):
        return "ZeroShotImgClassificationReply"

    pack_response_to_proto = single_string_response_to_proto

    def pack_request_to_proto(self, request_dict, **query_kwargs):
        return modelresponse_pb2.ZeroShotImgClassificationRequest(
            image=request_dict["image"],
            candidate_labels=request_dict["candidate_labels"],
            query_kwargs=kwarg_dict_to_proto(query_kwargs),
        )

    def unpack_request_from_proto(self, request):
        kwargs = unpack_proto_query_kwargs(request.query_kwargs)
        args = (request.image, request.candidate_labels)
        return args, kwargs

    def run_inference(self, inference_pipeline, args, kwargs):
        image, candidate_labels = args
        return inference_pipeline(image, candidate_labels=candidate_labels, **kwargs)


class InpaintingMethods(Text2ImgMethods):
    @property
    def method(self):
        return "InpaintingReply"

    def run_inference(self, inference_pipeline, args, kwargs):
        prompt, image, mask_image, negative_prompt = args
        return inference_pipeline(prompt=prompt,
                                  image=image,
                                  mask_image=mask_image,
                                  negative_prompt=negative_prompt,
                                  **kwargs)

    def pack_request_to_proto(self, request_dict, **query_kwargs):
        prompt = request_dict["prompt"]
        prompt = prompt if isinstance(prompt, list) else [prompt]
        negative_prompt = request_dict.get("negative_prompt", [""] * len(prompt))
        negative_prompt = negative_prompt if isinstance(negative_prompt,
                                                        list) else [negative_prompt]
        image = request_dict["image"] if isinstance(request_dict["image"],
                                                    list) else [request_dict["image"]]
        mask_image = request_dict["mask_image"] if isinstance(
            request_dict["mask_image"],
            list) else [request_dict["mask_image"]]

        return modelresponse_pb2.InpaintingRequest(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            query_kwargs=kwarg_dict_to_proto(query_kwargs),
        )

    def unpack_request_from_proto(self, request):
        kwargs = unpack_proto_query_kwargs(request.query_kwargs)

        image = [convert_bytes_to_pil_image(img) for img in request.image]
        mask_image = [
            convert_bytes_to_pil_image(mask_image) for mask_image in request.mask_image
        ]

        args = (list(request.prompt), image, mask_image, list(request.negative_prompt))
        return args, kwargs


GRPC_METHOD_TABLE = {
    TaskType.TEXT_GENERATION: TextGenerationMethods(),
    TaskType.TEXT_CLASSIFICATION: TextClassificationMethods(),
    TaskType.QUESTION_ANSWERING: QuestionAnsweringMethods(),
    TaskType.FILL_MASK: FillMaskMethods(),
    TaskType.TOKEN_CLASSIFICATION: TokenClassificationMethods(),
    TaskType.CONVERSATIONAL: ConversationalMethods(),
    TaskType.TEXT2IMG: Text2ImgMethods(),
    TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION: ZeroShotImgClassificationMethods(),
    TaskType.INPAINTING: InpaintingMethods(),
}
