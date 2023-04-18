# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from transformers import Conversation

from mii.constants import Tasks
from mii.grpc_related.proto import modelresponse_pb2
from mii.utils import kwarg_dict_to_proto, unpack_proto_query_kwargs
from mii.models.utils import ImageResponse


def single_string_request_to_proto(request_dict, **query_kwargs):
    return modelresponse_pb2.SingleStringRequest(
        request=request_dict['query'],
        query_kwargs=kwarg_dict_to_proto(query_kwargs))


def single_string_response_to_proto(response, time_taken, model_time_taken):
    return modelresponse_pb2.SingleStringReply(response=f"{response}",
                                               time_taken=time_taken,
                                               model_time_taken=model_time_taken)


def multi_string_request_to_proto(request_dict, **query_kwargs):
    return modelresponse_pb2.MultiStringRequest(
        request=request_dict['query'] if isinstance(request_dict['query'],
                                                    list) else [request_dict['query']],
        query_kwargs=kwarg_dict_to_proto(query_kwargs))


def proto_request_to_single_input(request):
    args = (request.request, )
    kwargs = unpack_proto_query_kwargs(request.query_kwargs)
    return args, kwargs


def proto_request_to_list(request):
    args = ([r for r in request.request], )
    kwargs = unpack_proto_query_kwargs(request.query_kwargs)
    return args, kwargs


def text_generation_pack_response_to_proto(response, time_taken, model_time_taken):
    text_responses = []
    for response in response:
        text = response[0]['generated_text']
        text_responses.append(text)

    return modelresponse_pb2.MultiStringReply(response=text_responses,
                                              time_taken=time_taken,
                                              model_time_taken=model_time_taken)


def question_answering_unpack_request_from_proto(request):
    kwargs = unpack_proto_query_kwargs(request.query_kwargs)
    kwargs["question"] = request.question
    kwargs["context"] = request.context
    args = ()
    return args, kwargs


def question_answering_pack_request_to_proto(request_dict, **query_kwargs):
    return modelresponse_pb2.QARequest(question=request_dict['question'],
                                       context=request_dict['context'],
                                       query_kwargs=kwarg_dict_to_proto(query_kwargs))


def conversational_pack_request_to_proto(request_dict, **query_kwargs):
    return modelresponse_pb2.ConversationRequest(
        text=request_dict['text'],
        conversation_id=request_dict['conversation_id']
        if 'conversation_id' in request_dict else None,
        past_user_inputs=request_dict['past_user_inputs'],
        generated_responses=request_dict['generated_responses'],
        query_kwargs=kwarg_dict_to_proto(query_kwargs))


def conversational_unpack_request_from_proto(request):
    kwargs = unpack_proto_query_kwargs(request.query_kwargs)
    conv = Conversation(text=request.text,
                        conversation_id=request.conversation_id,
                        past_user_inputs=request.past_user_inputs,
                        generated_responses=request.generated_responses,
                        **kwargs)
    args = (conv, )
    kwargs = {}
    return args, kwargs


def conversational_pack_response_to_proto(conv, time_taken, model_time_taken):
    return modelresponse_pb2.ConversationReply(
        conversation_id=conv.uuid,
        past_user_inputs=conv.past_user_inputs,
        generated_responses=conv.generated_responses,
        time_taken=time_taken,
        model_time_taken=model_time_taken)


def text2img_pack_response_to_proto(response, time_taken, model_time_taken):
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


def text2img_unpack_response_from_proto(response):
    return ImageResponse(response)


GRPC_METHOD_TABLE = {
    Tasks.TEXT_GENERATION: {
        "method": "GeneratorReply",
        "pack_request_to_proto": multi_string_request_to_proto,
        "unpack_request_from_proto": proto_request_to_list,
        "pack_response_to_proto": text_generation_pack_response_to_proto
    },
    Tasks.TEXT_CLASSIFICATION: {
        "method": "ClassificationReply",
        "pack_request_to_proto": single_string_request_to_proto,
        "unpack_request_from_proto": proto_request_to_single_input,
        "pack_response_to_proto": single_string_response_to_proto
    },
    Tasks.QUESTION_ANSWERING: {
        "method": "QuestionAndAnswerReply",
        "pack_request_to_proto": question_answering_pack_request_to_proto,
        "unpack_request_from_proto": question_answering_unpack_request_from_proto,
        "pack_response_to_proto": single_string_response_to_proto
    },
    Tasks.FILL_MASK: {
        "method": "FillMaskReply",
        "pack_request_to_proto": single_string_request_to_proto,
        "unpack_request_from_proto": proto_request_to_single_input,
        "pack_response_to_proto": single_string_response_to_proto
    },
    Tasks.TOKEN_CLASSIFICATION: {
        "method": "TokenClassificationReply",
        "pack_request_to_proto": single_string_request_to_proto,
        "unpack_request_from_proto": proto_request_to_single_input,
        "pack_response_to_proto": single_string_response_to_proto
    },
    Tasks.CONVERSATIONAL: {
        "method": "ConversationalReply",
        "pack_request_to_proto": conversational_pack_request_to_proto,
        "unpack_request_from_proto": conversational_unpack_request_from_proto,
        "pack_response_to_proto": conversational_pack_response_to_proto
    },
    Tasks.TEXT2IMG: {
        "method": "Txt2ImgReply",
        "pack_request_to_proto": multi_string_request_to_proto,
        "unpack_request_from_proto": proto_request_to_list,
        "pack_response_to_proto": text2img_pack_response_to_proto,
        "unpack_response_from_proto": text2img_unpack_response_from_proto
    }
}
