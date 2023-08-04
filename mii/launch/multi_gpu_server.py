# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import argparse
import mii
import torch
from mii import MIIConfig, LoadBalancerConfig

from mii.models.load_models import load_models
from mii.grpc_related.modelresponse_server import serve_inference, serve_load_balancing
from mii.grpc_related.restful_gateway import RestfulGatewayThread
from .utils import decode_config_from_str
from deepspeed.runtime.utils import see_memory_usage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--deployment-name", type=str, help="deployment name")
    parser.add_argument("-t", "--task-name", type=str, help="task name")
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-d", "--model-path", type=str, help="path to model")
    parser.add_argument('-b', '--provider', type=str, help="model provider")
    parser.add_argument("-o",
                        "--ds-optimize",
                        action='store_true',
                        help="Enable DeepSpeed")
    parser.add_argument("-z",
                        "--ds-zero",
                        action='store_true',
                        help="Enable DeepSpeed ZeRO")
    parser.add_argument("--ds-config", type=str, help="path to DeepSpeed ZeRO config")

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="base server port, each rank will have unique port based on this value")
    parser.add_argument("-c", "--config", type=str, help="base64 encoded mii config")
    parser.add_argument("--load-balancer",
                        type=str,
                        default=None,
                        help="base64 encoded load balancer config")
    parser.add_argument("-r",
                        "--restful-gateway",
                        action='store_true',
                        help="launch restful api gateway")

    args = parser.parse_args()

    # de-serialize config object
    config_dict = decode_config_from_str(args.config)
    # convert dict -> mii config
    mii_config = MIIConfig(**config_dict)

    if args.restful_gateway:
        print(f"Starting RESTful API gateway on port: {mii_config.restful_api_port}")
        gateway_thread = RestfulGatewayThread(args.deployment_name,
                                              args.task_name,
                                              mii_config)
        stop_event = gateway_thread.get_stop_event()
        gateway_thread.start()
        stop_event.wait()

    elif args.load_balancer is None:
        provider = mii.constants.MODEL_PROVIDER_MAP.get(args.provider, None)
        assert provider is not None, f"Unknown model provider: {args.provider}"

        assert args.port is not None, "port is required for inference server"
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        port = args.port + local_rank

        inference_pipeline = load_models(task_name=args.task_name,
                                         model_name=args.model,
                                         model_path=args.model_path,
                                         ds_optimize=args.ds_optimize,
                                         ds_zero=args.ds_zero,
                                         ds_config_path=args.ds_config,
                                         provider=provider,
                                         mii_config=mii_config)
        see_memory_usage("BEFORE SWITCH", force=True)
        inference_pipeline.model.to(torch.device("cpu"))
        see_memory_usage("AFTER SWITCH", force=True)
        print(f"Starting server on port: {port}")
        serve_inference(inference_pipeline, port)
    else:
        lb_config_dict = decode_config_from_str(args.load_balancer)
        lb_config = LoadBalancerConfig(**lb_config_dict)

        print(f"Starting load balancer on port: {lb_config.port}")
        serve_load_balancing(args.task_name, lb_config)


if __name__ == "__main__":
    # python -m mii.launch.multi_gpu_server
    main()
