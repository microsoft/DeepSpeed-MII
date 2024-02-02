# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# Standard library imports
import json
import grpc
import argparse

# Third-party imports
import uvicorn
import mii
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from mii.grpc_related.proto.modelresponse_pb2_grpc import ModelResponseStub
from mii.grpc_related.proto import modelresponse_pb2
from mii.utils import kwarg_dict_to_proto

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

    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]

    # Set up the generation arguments
    generate_args = {"ignore_eos": False, "do_sample": True, "return_full_text": False}

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

    # Streaming case
    if request.stream:
        return JSONResponse({"error": "Streaming is not yet supported."},
                            status_code=400)
        # async def StreamResults() -> AsyncGenerator[bytes, None]:
        #     # Send an empty chunk to start the stream and prevent timeout
        #     yield ""
        #     async for response_chunk in stub.GeneratorReplyStream(requestData):
        #         # Send the response chunk
        #         responses = [obj.response for obj in response_chunk.response]
        #         dataOut = {"text": responses}
        #         yield f"data: {json.dumps(dataOut)}\n\n"
        #     yield f"data: [DONE]\n\n"
        # return StreamingResponse(StreamResults(), media_type="text/event-stream")

    # Non-streaming case
    responseData = await stub.GeneratorReply(requestData)
    responses = [obj.response for obj in responseData.response]
    result = {"text": responses}
    return JSONResponse(result)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return JSONResponse({"status": "ok"}, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DeepSpeed-MII Simple Text Generation RESRful API Server")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help=
        "model name or path to model directory (defaults to mistralai/Mistral-7B-Instruct-v0.1)"
    )
    parser.add_argument(
        '--deployment-name',
        type=str,
        default="deepspeed-mii",
        help=
        'A unique identifying string for the persistent model (defaults to f"deepspeed-mii")'
    )
    parser.add_argument("--load-balancer",
                        type=str,
                        default=None,
                        help="load balancer address (defaults to None)")
    parser.add_argument("--max-length",
                        type=int,
                        default=32768,
                        help="maximum token length (defaults to 32768)")
    parser.add_argument("--host",
                        type=str,
                        default="0.0.0.0",
                        help="host address (defaults to 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="port (defaults to 8000)")
    parser.add_argument(
        "--allow-credentials",
        action="store_true",\
        help="allow credentials"
    )
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument(
        '--max_length',
        type=int,
        default=None,
        help=
        'Sets the default maximum token length for the prompt + response (defaults to maximum sequence length in model config)'
    )
    parser.add_argument('--tensor-parallel',
                        type=int,
                        default=1,
                        help='Number of GPUs to split the model across (defaults to 1)')
    parser.add_argument('--replica-num',
                        type=int,
                        default=1,
                        help='The number of model replicas to stand up (defaults to 1)')

    args = parser.parse_args()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    # Check if a load balancer is specified else start the DeepSpeed-MII instance
    if args.load_balancer is not None:
        # Set the load balancer
        load_balancer = args.load_balancer
    else:
        # Initialize the DeepSpeed-MII instance
        mii.serve(args.model,
                  deployment_name=args.deployment_name,
                  tensor_parallel=args.tensor_parallel,
                  replica_num=args.replica_num,
                  max_length=args.max_length)

    # Start the server
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=300)
