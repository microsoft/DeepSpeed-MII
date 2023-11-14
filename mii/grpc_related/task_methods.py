# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from google.protobuf.message import Message

from mii.batching.data_classes import Response, ResponseBatch
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


class TaskMethods(ABC):
    @property
    @abstractmethod
    def method(self):
        ...

    @abstractmethod
    def pack_request_to_proto(self, request, **query_kwargs):
        ...

    @abstractmethod
    def unpack_request_from_proto(self, proto_request):
        ...

    @abstractmethod
    def pack_response_to_proto(self, response):
        ...

    @abstractmethod
    def unpack_response_from_proto(self, proto_response):
        ...


class TextGenerationMethods(TaskMethods):
    @property
    def method(self):
        return "GeneratorReply"

    @property
    def method_stream_out(self):
        return "GeneratorReplyStream"

    def pack_request_to_proto(self,
                              prompts: List[str],
                              **query_kwargs: Dict[str,
                                                   Any]) -> Message:
        proto_request = modelresponse_pb2.MultiStringRequest(
            request=prompts,
            query_kwargs=kwarg_dict_to_proto(query_kwargs),
        )
        return proto_request

    def unpack_request_from_proto(self,
                                  proto_request: Message) -> Tuple[List[str],
                                                                   Dict[str,
                                                                        Any]]:
        prompts = [r for r in proto_request.request]
        kwargs = unpack_proto_query_kwargs(proto_request.query_kwargs)
        return prompts, kwargs

    def pack_response_to_proto(self, responses: ResponseBatch) -> Message:
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
            time_taken=-1,
            model_time_taken=-1,
        )

    def unpack_response_from_proto(self, response: Message) -> ResponseBatch:
        response_batch = ResponseBatch()
        for i, r in enumerate(response.response):
            response_batch.append(
                Response(
                    generated_text=r,
                    prompt_length=response.details[i].prompt_tokens,
                    generated_length=response.details[i].generated_tokens,
                    finish_reason=response.details[i].finish_reason,
                ))
        return response_batch


TASK_METHODS_DICT = {
    TaskType.TEXT_GENERATION: TextGenerationMethods(),
}
