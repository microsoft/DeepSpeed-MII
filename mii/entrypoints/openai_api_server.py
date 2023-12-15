import grpc
import argparse
import json
import os
from typing import AsyncGenerator, Optional, Dict, List, Union
from transformers import AutoTokenizer
import codecs

import fastapi
from fastapi import FastAPI, Depends, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

import shortuuid
import tiktoken
import uvicorn
import mii
from mii.grpc_related.proto.modelresponse_pb2_grpc import ModelResponseStub
from mii.grpc_related.proto import modelresponse_pb2
from mii.utils import kwarg_dict_to_proto
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
tokenizer = None
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

def countTokens(prompt: Union[str, list[str]]) -> int:
    if isinstance(prompt, str):
        prompt = [prompt]

    total_tokens = 0
    for p in prompt:
        total_tokens += len(tokenizer(p).input_ids)

    return total_tokens

def load_chat_template(args, tokenizer):
    if args.chat_template is not None:
        try:
            with open(args.chat_template, "r") as f:
                chat_template = f.read()
        except OSError:
            # If opening a file fails, set chat template to be args to
            # ensure we decode so our escape are interpreted correctly
            chat_template = codecs.decode(args.chat_template, "unicode_escape")

        tokenizer.chat_template = chat_template
        print(f"Chat template loaded from {args.chat_template}.")
    elif tokenizer.chat_template is not None:
        print(f"Chat template loaded from tokenizer.")
    else:
        # throw a warning if no chat template is provided
        print("WARNING: No chat template provided. chat completion won't work.")

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

    finalPrompt = tokenizer.apply_chat_template(
        conversation=request.messages,
        tokenize=False,
        add_generation_prompt=request.add_generation_prompt)

    prompts = [finalPrompt for _ in range(request.n)]

    requestData = modelresponse_pb2.MultiStringRequest(
        request=prompts,
        query_kwargs=kwarg_dict_to_proto(generate_args),
    )

    id = f"chatcmpl-{shortuuid.random()}"

    response_role = "assistant"

    if request.add_generation_prompt:
        response_role = app_settings.response_role

    # Streaming case
    if request.stream:
        async def StreamResults() -> AsyncGenerator[bytes, None]:
            # First chunk with role
            firstChoices = []
            for _ in range(request.n):
                firstChoice = ChatCompletionResponseStreamChoice(
                    index=len(firstChoices),
                    delta=DeltaMessage(role=response_role),
                    finish_reason=None,
                )
                firstChoices.append(firstChoice)

            chunk = ChatCompletionStreamResponse(
                id=id, choices=firstChoices, model=app_settings.model_id
            )
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
            async for response_chunk in stub.GeneratorReplyStream(requestData):
                streamChoices = []

                for c in response_chunk.response:
                    choice = ChatCompletionResponseStreamChoice(
                        index=len(streamChoices),
                        delta=DeltaMessage(content=c.response),
                        finish_reason=None if c.finish_reason == "none" else c.finish_reason,
                    )
                    streamChoices.append(choice)

                chunk = ChatCompletionStreamResponse(
                    id=id, choices=streamChoices, model=app_settings.model_id
                )
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(StreamResults(), media_type="text/event-stream")
    
    # Non-streaming case
    responseData = await stub.GeneratorReply(requestData)
    choices = []

    for c in responseData.response:
        choice = ChatCompletionResponseChoice(
            index=len(choices),
            message=ChatMessage(role=response_role, content=c.response),
            finish_reason=None if c.finish_reason == "none" else c.finish_reason,
        )
        choices.append(choice)

    prompt_tokens = countTokens([message['content'] for message in request.messages if 'content' in message]) * request.n
    completion_tokens = countTokens([r.response for r in responseData.response])

    return ChatCompletionResponse(model=app_settings.model_id, choices=choices, usage=UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    ))

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

    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]

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
    requestData = modelresponse_pb2.MultiStringRequest(
        request=request.prompt,
        query_kwargs=kwarg_dict_to_proto(generate_args),
    )
    id = f"cmpl-{shortuuid.random()}"
    # Streaming case
    if request.stream:
        async def StreamResults() -> AsyncGenerator[bytes, None]:
            # Send an empty chunk to start the stream and prevent timeout
            yield ""
            async for response_chunk in stub.GeneratorReplyStream(requestData):
                streamChoices = []

                for c in response_chunk.response:
                    choice = CompletionResponseStreamChoice(
                        index=len(streamChoices),
                        text=c.response,
                        logprobs=None,
                        finish_reason=None if c.finish_reason == "none" else c.finish_reason,
                    )
                    streamChoices.append(choice)

                chunk = CompletionStreamResponse(
                    id=id,
                    object="text_completion",
                    choices=streamChoices,
                    model=app_settings.model_id,
                )
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(StreamResults(), media_type="text/event-stream")

    # Non-streaming case
    responseData = await stub.GeneratorReply(requestData)
    choices = []

    for c in responseData.response:
        choice = CompletionResponseChoice(
            index=len(choices),
            text=c.response,
            logprobs=None,
            finish_reason=None if c.finish_reason == "none" else c.finish_reason,
        )
        choices.append(choice)

    prompt_tokens = countTokens(request.prompt)
    completion_tokens = countTokens([r.response for r in responseData.response])

    return CompletionResponse(
        model=app_settings.model_id, choices=choices, usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
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
    parser.add_argument("--chat-template", type=str, default=None)
    parser.add_argument("--response-role", type=str, default="assistant")
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

    if args.api_keys is not None:
        app_settings.api_keys = args.api_keys

    # Check if a load balancer is specified else start the DeepSpeed-MII instance
    if args.load_balancer is not None:
        # Set the load balancer
        load_balancer = args.load_balancer
    else:
        # Initialize the DeepSpeed-MII instance
        mii.serve(args.model, deployment_name=args.deployment_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_chat_template(args, tokenizer)

    if args.response_role is not None:
        app_settings.response_role = args.response_role

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