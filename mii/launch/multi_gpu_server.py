import sys, os
import argparse
from mii.models.load_models import load_generator_models
from mii.grpc_related.modelresponse_server import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-d", "--model-path", type=str, help="path to model")
    parser.add_argument("-p", "--port", type=int, help="base server port, each rank will have unique port based on this value")
    args = parser.parse_args()

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    print(local_rank)
    port = args.port + local_rank
    generator = load_generator_models(args.model, args.model_path)
    print(generator("Test product is ",  do_sample=True, min_length=50))
    serve(generator, port)


if __name__ == "__main__":
    # python -m mii.launch.multi_gpu_server
    main()