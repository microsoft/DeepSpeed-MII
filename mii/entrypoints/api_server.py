# Standard library imports
import json
import time
import grpc
import asyncio
import argparse
import threading
from queue import Queue
from typing import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import fastapi
import uvicorn
import mii
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from mii.grpc_related.proto.modelresponse_pb2_grpc import ModelResponseStub
from mii.grpc_related.task_methods import multi_string_request_to_proto

# Local module imports
from .data_models import CompletionRequest

app = FastAPI()
load_balancer = "localhost:50050"

@app.post("/generate")
async def generate(request: CompletionRequest) -> Response:
    # TODO: Add support for multiple stop tokens, as for now only one is supported
    # Check if stop token is a list
    if request.stop is not None and isinstance(request.stop, list):
        request.stop = request.stop[0]

    # Set defaults
    if request.max_tokens is None:
        request.max_tokens = 128

    if request.stream is None:
        request.stream = False

    if request.prompt is None:
        return JSONResponse({"error": "Prompt is required."}, status_code=400)

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

    # Streaming case
    if request.stream:
        async def StreamResults() -> AsyncGenerator[bytes, None]:
            # Send an empty chunk to start the stream and prevent timeout
            yield ""
            async for response_chunk in stub.GeneratorReplyStream(requestData):
                # Send the response chunk
                dataOut = {"text": response_chunk.response[0]}
                yield f"data: {json.dumps(dataOut)}\n\n"
            yield f"data: [DONE]\n\n"
        return StreamingResponse(StreamResults(), media_type="text/event-stream")

    # Non-streaming case
    responseData = await stub.GeneratorReply(requestData)
    result = {"text": responseData.response[0]}
    return JSONResponse(result)

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return JSONResponse({"status": "ok"}, status_code=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    # Set the deployment name
    deployment_name = args.deployment_name

    # Check if a load balancer is specified else start the DeepSpeed-MII instance
    if args.load_balancer is not None:
        # Set the load balancer
        load_balancer = args.load_balancer
    else:
        # Initialize the DeepSpeed-MII instance
        mii.serve(args.model, deployment_name=deployment_name)

    # Start the server
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=300)