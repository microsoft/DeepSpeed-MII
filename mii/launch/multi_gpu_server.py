'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import sys, os
import argparse
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
    parser.add_argument("-z",
                        "--ds-zero",
                        action='store_true',
                        help="Enable DeepSpeed ZeRO")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="base server port, each rank will have unique port based on this value")
    args = parser.parse_args()

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    print(local_rank)
    port = args.port + local_rank
    inference_pipeline = load_models(args.task_name,
                                     args.model,
                                     args.model_path,
                                     args.ds_optimize,
                                     args.ds_zero)
    #print(inference("Test product is ", do_sample=True, min_length=50))
    serve(inference_pipeline, port)


if __name__ == "__main__":
    # python -m mii.launch.multi_gpu_server
    main()
