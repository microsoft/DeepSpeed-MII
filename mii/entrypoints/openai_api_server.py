import grpc
import argparse
import json
import os
from typing import AsyncGenerator, Optional, Dict, List

import fastapi
from fastapi import FastAPI, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

import shortuuid
import tiktoken
import uvicorn
import mii
from mii.grpc_related.proto.modelresponse_pb2_grpc import ModelResponseStub
from mii.grpc_related.task_methods import multi_string_request_to_proto
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from fastchat.constants import ErrorCode

from .data_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
    AppSettings,
)

app = FastAPI()
load_balancer = "localhost:50050"
app_settings = AppSettings()
get_bearer_token = HTTPBearer(auto_error=False)

async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))

@app.get("/v1/models", dependencies=[Depends(check_api_key)])
async def show_available_models():
    models.sort()
    # TODO: return real model permission details
    model_cards = []
    model_cards.append(ModelCard(id=app_settings.model_id, root=app_settings.model_id, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):
    if request.model != app_settings.model_id:
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"Model \"{request.model}\" not found. Please use \"{app_settings.model_id}\".",
        )

    if request.stop is not None and isinstance(request.stop, list):
        request.stop = request.stop[0]

    # Set defaults
    if request.max_tokens is None:
        request.max_tokens = 128

    if request.stream is None:
        request.stream = False

    if request.messages is None:
        return create_error_response(
            ErrorCode.PARAM_REQUIRED,
            "messages is required.",
        )

    # Set up the generation arguments
    generate_args = {
        "ignore_eos": False,
        "do_sample": True,
        "return_full_text": False
    }

    if request.min_tokens is not None:
        generate_args["min_new_tokens"] = request.min_tokens
    
    if request.max_tokens is not None:
        generate_args["max_new_tokens"] = request.max_tokens
    
    if request.top_p is not None:
        generate_args["top_p"] = request.top_p

    if request.top_k is not None:
        generate_args["top_k"] = request.top_k

    if request.temperature is not None:
        generate_args["temperature"] = request.temperature
    
    if request.stop is not None:
        generate_args["stop"] = request.stop    

    if request.stream:
        generate_args["stream"] = True

    channel = grpc.aio.insecure_channel(load_balancer)
    stub = ModelResponseStub(channel)

    conversation = get_conversation_template(app_settings.model_id)
    conversation = Conversation(
        name=conversation.name,
        system_template=conversation.system_template,
        system_message=conversation.system_message,
        roles=conversation.roles,
        messages=list(conversation.messages),  # prevent in-place modification
        offset=conversation.offset,
        sep_style=SeparatorStyle(conversation.sep_style),
        sep=conversation.sep,
        sep2=conversation.sep2,
        stop_str=conversation.stop_str,
        stop_token_ids=conversation.stop_token_ids,
    )

    if isinstance(request.messages, str):
        return create_error_response(
            ErrorCode.PARAM_TYPE_ERROR,
            "messages must be a list of objects.",
        )
    else:
        for message in request.messages:
            role = message["role"]
            if role == "system":
                conversation.system_message = message["content"]
            elif role == "user":
                conversation.append_message(conversation.roles[0], message["content"])
            elif role == "assistant":
                conversation.append_message(conversation.roles[1], message["content"])
            else:
                return create_error_response(
                    ErrorCode.PARAM_TYPE_ERROR,
                    "role must be one of 'user', 'system', 'assistant'.",
                )

    # Add a blank message for the assistant.
    conversation.append_message(conversation.roles[1], None)
    finalPrompt = conversation.get_prompt()

    requestData = multi_string_request_to_proto(self=None, request_dict={"query": finalPrompt}, **generate_args)
    id = f"chatcmpl-{shortuuid.random()}"

    # Streaming case
    if request.stream:
        async def StreamResults() -> AsyncGenerator[bytes, None]:
            # First chunk with role
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=id, choices=[choice_data], model=app_settings.model_id
            )
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
            async for response_chunk in stub.GeneratorReplyStream(requestData):
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=response_chunk.response[0]),
                    finish_reason=convert_reason(response_chunk.details[0].finish_reason),
                )
                chunk = ChatCompletionStreamResponse(
                    id=id, choices=[choice_data], model=app_settings.model_id
                )
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(StreamResults(), media_type="text/event-stream")
    
    # Non-streaming case
    responseData = await stub.GeneratorReply(requestData)
    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=responseData.response[0]),
        finish_reason=convert_reason(responseData.details[0].finish_reason),
    )
    return ChatCompletionResponse(model=app_settings.model_id, choices=[choice], usage=UsageInfo())

@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
async def create_completion(request: CompletionRequest):
    if request.model != app_settings.model_id:
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"Model {request.model} not found.",
        )

    if request.stop is not None and isinstance(request.stop, list):
        request.stop = request.stop[0]

    # Set defaults
    if request.max_tokens is None:
        request.max_tokens = 128

    if request.stream is None:
        request.stream = False

    if request.prompt is None:
        return create_error_response(
            ErrorCode.PARAM_REQUIRED,
            "Prompt is required.",
        )

    # Set up the generation arguments
    generate_args = {
        "ignore_eos": False,
        "do_sample": True,
        "return_full_text": False
    }

    # Set optional generation arguments
    if request.max_length is not None:
        generate_args["max_length"] = request.max_length
    
    if request.min_tokens is not None:
        generate_args["min_new_tokens"] = request.min_tokens
    
    if request.max_tokens is not None:
        generate_args["max_new_tokens"] = request.max_tokens
    
    if request.top_p is not None:
        generate_args["top_p"] = request.top_p

    if request.top_k is not None:
        generate_args["top_k"] = request.top_k

    if request.temperature is not None:
        generate_args["temperature"] = request.temperature
    
    if request.stop is not None:
        generate_args["stop"] = request.stop    

    if request.stream:
        generate_args["stream"] = True

    channel = grpc.aio.insecure_channel(load_balancer)
    stub = ModelResponseStub(channel)
    requestData = multi_string_request_to_proto(self=None, request_dict={"query": request.prompt}, **generate_args)
    id = f"cmpl-{shortuuid.random()}"
    # Streaming case
    if request.stream:
        async def StreamResults() -> AsyncGenerator[bytes, None]:
            # Send an empty chunk to start the stream and prevent timeout
            yield ""
            async for response_chunk in stub.GeneratorReplyStream(requestData):
                choice_data = CompletionResponseStreamChoice(
                    index=0,
                    text=response_chunk.response[0],
                    logprobs=None,
                    finish_reason=convert_reason(response_chunk.details[0].finish_reason),
                )
                chunk = CompletionStreamResponse(
                    id=id,
                    object="text_completion",
                    choices=[choice_data],
                    model=app_settings.model_id,
                )
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(StreamResults(), media_type="text/event-stream")

    # Non-streaming case
    responseData = await stub.GeneratorReply(requestData)
    choice = CompletionResponseChoice(
        index=0,
        text=responseData.response[0],
        logprobs=None,
        finish_reason=convert_reason(responseData.details[0].finish_reason),
    )
    return CompletionResponse(
        model=app_settings.model_id, choices=[choice], usage=UsageInfo()
    )

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return JSONResponse({"status": "ok"}, status_code=200)

def convert_reason(finish_reason):
    if finish_reason == "GenerationFinishReason.NONE":
        return None
    elif finish_reason == "GenerationFinishReason.LENGTH":
        return "length"
    elif finish_reason == "GenerationFinishReason.STOP":
        return "stop"
    else:
        return finish_reason

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI-Compatible RESTful API server.")
    parser.add_argument("--model", type=str)
    parser.add_argument("--deployment-name", type=str, default="deepspeed-mii")
    parser.add_argument("--load-balancer", type=str)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    parser.add_argument("--allowed-origins", type=json.loads, default=["*"], help="allowed origins")
    parser.add_argument("--allowed-methods", type=json.loads, default=["*"], help="allowed methods")
    parser.add_argument("--allowed-headers", type=json.loads, default=["*"], help="allowed headers")
    parser.add_argument("--api-keys", type=lambda s: s.split(","), help="Optional list of comma separated API keys")
    parser.add_argument("--ssl", action="store_true", required=False, default=False, help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.")
    args = parser.parse_args()

    # Set the deployment name
    deployment_name = args.deployment_name

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    app_settings.model_id = args.model

    # Check if a load balancer is specified else start the DeepSpeed-MII instance
    if args.load_balancer is not None:
        # Set the load balancer
        load_balancer = args.load_balancer
    else:
        # Initialize the DeepSpeed-MII instance
        mii.serve(args.model, deployment_name=args.deployment_name)

    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
            timeout_keep_alive=300
        )
    else:
        uvicorn.run(app,
            host=args.host,
            port=args.port,
            log_level="info",
            timeout_keep_alive=300
        )