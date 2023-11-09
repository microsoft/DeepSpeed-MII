# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod

from mii.constants import TaskType
from mii.grpc_related.proto import modelresponse_pb2
from mii.utils import kwarg_dict_to_proto, unpack_proto_query_kwargs


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


def proto_request_to_list(self, request):
    prompts = [r for r in request.request]
    kwargs = unpack_proto_query_kwargs(request.query_kwargs)
    return prompts, kwargs


class TaskMethods(ABC):
    @property
    @abstractmethod
    def method(self):
        ...

    def pack_request_to_proto(self, request_dict, **query_kwargs):
        return request_dict, query_kwargs

    def unpack_request_from_proto(self, request):
        return request

    def pack_response_to_proto(self, response, time_taken, model_time_taken):
        return response, time_taken, model_time_taken

    def unpack_response_from_proto(self, response):
        return response


class TextGenerationMethods(TaskMethods):
    @property
    def method(self):
        return "GeneratorReply"

    @property
    def method_stream_out(self):
        return "GeneratorReplyStream"

    pack_request_to_proto = multi_string_request_to_proto
    unpack_request_from_proto = proto_request_to_list

    def pack_response_to_proto(self, responses, time_taken, model_time_taken):
        text_responses = []
        details = []

        # Response a nested list of dicts
        # [Sample, 1, Dict]
        for response in responses:
            text = response.generated_text
            text_responses.append(text)
            details.append(
                modelresponse_pb2.GenerationDetails(
                    finish_reason=str(response.finish_reason),
                    prompt_tokens=response.prompt_length,
                    generated_tokens=response.generated_length))

        return modelresponse_pb2.GenerationReply(
            response=text_responses,
            indices=[0],
            details=details,
            time_taken=time_taken,
            model_time_taken=model_time_taken,
        )


TASK_METHODS_DICT = {
    TaskType.TEXT_GENERATION: TextGenerationMethods(),
}
