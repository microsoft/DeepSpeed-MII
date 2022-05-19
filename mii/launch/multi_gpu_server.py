'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import sys, os
import argparse
import base64
import json

from mii import MIIConfig
from mii.models.load_models import load_models
from mii.grpc_related.modelresponse_server import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task-name", type=str, help="task name")
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-d", "--model-path", type=str, help="path to model")
    parser.add_argument("-o",
                        "--ds-optimize",
                        action='store_true',
                        help="Enable DeepSpeed")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="base server port, each rank will have unique port based on this value")
    parser.add_argument("-c", "--config", type=str, help="base64 encoded mii config")
    args = parser.parse_args()

    # de-serialize config object
    # str -> bytes
    b64_bytes = args.config.encode()
    # decode b64 bytes -> json bytes
    config_bytes = base64.urlsafe_b64decode(b64_bytes)
    # convert json bytes -> str -> dict
    config_dict = json.loads(config_bytes.decode())
    # convert dict -> mii config
    mii_config = MIIConfig(**config_dict)

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    print(local_rank)
    port = args.port + local_rank
    inference_pipeline = load_models(args.task_name,
                                     args.model,
                                     args.model_path,
                                     args.ds_optimize,
                                     mii_config=mii_config)
    #print(inference("Test product is ", do_sample=True, min_length=50))
    serve(inference_pipeline, port)


if __name__ == "__main__":
    # python -m mii.launch.multi_gpu_server
    main()
