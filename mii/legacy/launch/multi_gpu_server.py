# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import argparse
import base64
import json

from mii.legacy.config import ModelConfig
from mii.legacy.models.load_models import load_models
from mii.legacy.grpc_related.modelresponse_server import serve_inference, serve_load_balancing
from mii.legacy.grpc_related.restful_gateway import RestfulGatewayThread


def b64_encoded_config(config_str):
    # str -> bytes
    b64_bytes = config_str.encode()
    # decode b64 bytes -> json bytes
    config_bytes = base64.urlsafe_b64decode(b64_bytes)
    # convert json bytes -> str -> dict
    config_dict = json.loads(config_bytes.decode())
    # return mii.ModelConfig object
    return ModelConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-name", type=str, help="Name of deployment")
    parser.add_argument(
        "--model-config",
        type=b64_encoded_config,
        help="base64 encoded model config",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=0,
        help="Port to user for DeepSpeed inference server.",
    )
    parser.add_argument("--load-balancer",
                        action="store_true",
                        help="Launch load balancer process.")
    parser.add_argument(
        "--load-balancer-port",
        type=int,
        default=0,
        help="Port to use for load balancer.",
    )
    parser.add_argument(
        "--restful-gateway",
        action="store_true",
        help="Launches restful gateway process.",
    )
    parser.add_argument(
        "--restful-gateway-port",
        type=int,
        default=0,
        help="Port to use for restful gateway.",
    )
    args = parser.parse_args()
    assert not (
        args.load_balancer and args.restful_gateway
    ), "Select only load-balancer OR restful-gateway."

    if args.restful_gateway:
        assert args.restful_gateway_port, "--restful-gateway-port must be provided."
        print(f"Starting RESTful API gateway on port: {args.restful_gateway_port}")
        gateway_thread = RestfulGatewayThread(
            deployment_name=args.deployment_name,
            task=args.model_config.task,
            lb_port=args.load_balancer_port,
            rest_port=args.restful_gateway_port,
        )
        stop_event = gateway_thread.get_stop_event()
        gateway_thread.start()
        stop_event.wait()

    elif args.load_balancer:
        assert args.load_balancer_port, "--load-balancer-port must be provided."
        print(f"Starting load balancer on port: {args.load_balancer_port}")
        serve_load_balancing(args.model_config, args.load_balancer_port)

    else:
        assert args.server_port, "--server-port must be provided."
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        port = args.server_port + local_rank

        inference_pipeline = load_models(args.model_config)

        print(f"Starting server on port: {port}")
        serve_inference(inference_pipeline, port)


if __name__ == "__main__":
    # python -m mii.launch.multi_gpu_server
    main()
