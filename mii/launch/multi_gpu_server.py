'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import argparse
import mii
from mii.models.load_models import load_models
from mii.grpc_related.modelresponse_server import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task-name", type=str, help="task name")
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-d", "--model-path", type=str, help="path to model")
    parser.add_argument('-b', '--provider', type=str, help="model provider")
    parser.add_argument("-o",
                        "--ds-optimize",
                        action='store_true',
                        help="Enable DeepSpeed")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="base server port, each rank will have unique port based on this value")
    args = parser.parse_args()

    provider = mii.constants.MODEL_PROVIDER_MAP.get(args.provider, None)
    assert provider is not None, f"Unknown model provider: {args.provider}"

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    port = args.port + local_rank

    inference_pipeline = load_models(task_name=args.task_name,
                                     model_name=args.model,
                                     model_path=args.model_path,
                                     ds_optimize=args.ds_optimize,
                                     provider=provider)
    serve(inference_pipeline, port)


if __name__ == "__main__":
    # python -m mii.launch.multi_gpu_server
    main()
