from cgitb import enable
import subprocess
import sys
import os
import time
import argparse
import csv

from statistics import mean
from get_hf_models import Model
import mii


def size_to_string(size, units=None, precision=2):
    if units is None:
        if size // 10**12 > 0:
            return str(round(size / 10**12, 2)) + " T"
        elif size // 10**9 > 0:
            return str(round(size / 10**9, 2)) + " G"
        elif size // 10**6 > 0:
            return str(round(size / 10**6, 2)) + " M"
        elif size // 10**3:
            return str(round(size / 10**3, 2)) + " K"
        else:
            return str(size)
    else:
        if units == "T":
            return str(round(size / 10.0**12, precision)) + " " + units
        elif units == "G":
            return str(round(size / 10.0**9, precision)) + " " + units
        elif units == "M":
            return str(round(size / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(size / 10.0**3, precision)) + " " + units
        else:
            return str(size)


def _deploy_model(model, mii_configs, enable_deepspeed=True):
    name = model.name
    type = model.type
    task = model.task
    mii.deploy(task,
               name,
               mii.DeploymentType.LOCAL,
               deployment_name=name + "_deployment",
               local_model_path=".cache/models/" + name,
               mii_configs=mii_configs,
               enable_deepspeed=enable_deepspeed)


def _kill_deployment(model, mii_configs={}):
    kill_cmd = [
        'pkill',
        '-f',
        '-9',
        "python",
    ]

    result = subprocess.Popen(kill_cmd)
    result.wait()

    if result.returncode > 0:
        sys.exit(result.returncode)
    print(f"Killed deployment {model.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark models")
    parser.add_argument("-f",
                        "--model_file",
                        type=str,
                        default="sampled_models_gpt2.json",
                        help="Path to file containing model list")
    parser.add_argument("-o",
                        "--output_file",
                        type=str,
                        default="bench_output_gpt2.csv",
                        help="Path to file containing benchmark output")
    parser.add_argument("-i",
                        "--model_index",
                        type=int,
                        default=0,
                        help="Index of model in the model_files to benchmark")
    parser.add_argument("--model_name",
                        type=str,
                        default=None,
                        help="Name of the model in the model_files to benchmark")
    parser.add_argument("-n",
                        "--num_iters",
                        type=int,
                        help="number of iterations to run",
                        default=20)
    parser.add_argument(
        "-t",
        "--data_type",
        type=str,
        default="fp32",
        help="data type used for inference, either fp32 or fp16, default is fp32")
    parser.add_argument("--disable_deepspeed", action='store_true')
    parser.add_argument("--reuse_output", action='store_true')

    args = parser.parse_args()

    model_file = args.model_file
    models = []
    if os.path.exists(model_file):
        with open(model_file) as f:
            data = f.read()
            models = Model.schema().loads(data, many=True)

    print(f"Populated {len(models)} models from file {model_file}")

    model_index = args.model_index
    model = models[model_index]
    output_file = args.output_file

    if args.model_name is not None:
        find = [m for m in models if m.name == args.model_name]
        print(find)
        if find:
            model = find[0]
            output_file = "test_" + (args.model_name).replace('/', '_')

    num_iters = args.num_iters
    enable_deepspeed = not args.disable_deepspeed
    data_type = 'fp32' if args.data_type == 'fp32' else 'fp16'

    def get_line(name, path):
        with open(path) as f:
            lines = f.readlines()
            for l in lines:
                if name in l:
                    n = l.split(",")[1]
                    if name == n.replace(" ", ""):
                        return l
        return None

    if os.path.exists(output_file) and args.reuse_output:
        line = get_line(model.name, output_file)

        if line and (enable_deepspeed or ((not enable_deepspeed) and "False" in line)):
            print(
                f"Skipping {model_index}: {model.name} as it already exists in {output_file}"
            )
            sys.exit(0)

    input = "DeepSpeed is the greatest"  # 5 tokens, deepspeed is counted as 2 tokens

    print(
        f"Benchmarking {model_index}: {model.name}, {model.type}, {model.task}, {size_to_string(model.size/4)} with enable_deepspeed={enable_deepspeed}"
    )

    _deploy_model(model,
                  mii_configs={
                      'dtype': data_type,
                      'enable_cuda_graph': True
                  },
                  enable_deepspeed=enable_deepspeed)

    time_takens = []
    for i in range(num_iters):
        if model.task == "conversational":
            generator = mii.mii_query_handle(model.name + "_deployment")
            result = generator.query({
                'text': input,
                'conversation_id': 3,
                'past_user_inputs': [],
                'generated_responses': []
            })
            if i < 1: continue  # warmup
            time_takens.append(result.time_taken)
        elif model.task == "question-answering":
            generator = mii.mii_query_handle(model.name + "_deployment")
            result = generator.query({
                'question': "What is the greatest",
                'context': "DeepSpeed is the greatest"
            })
            if i < 1: continue
            time_takens.append(result.time_taken)
        elif model.task == "fill-mask":
            generator = mii.mii_query_handle(model.name + "_deployment")
            if model.name in ["ufal/robeczech-base", "klue/roberta-large"]:
                mask = "[MASK]"
            elif model.name in [
                    "flaubert/flaubert_large_cased",
                    "flaubert/flaubert_base_uncased"
            ]:
                mask = "<special1>"
                input
            elif model.type == "bert":
                mask = "[MASK]"
            elif model.type == "roberta":
                mask = "<mask>"
            else:
                mask = "<mask>"
            input = "DeepSpeed is the " + mask
            result = generator.query({'query': input})
            if i < 1: continue
            time_takens.append(result.time_taken)
        else:
            generator = mii.mii_query_handle(model.name + "_deployment")
            result = generator.query({'query': input})
            if i < 1: continue  # warmup
            time_takens.append(result.time_taken)

    mean_time = mean(time_takens)
    print(f"mean time_taken: {mean_time}")

    if enable_deepspeed:
        if args.model_name is not None:
            with open(output_file, 'w') as f:
                f.write(
                    f"{model_index}, {model.name}, {model.type}, {size_to_string(model.size/4)}, {model.size}, {model.task}, {model.url}, {model.downloads}, {enable_deepspeed}, {mean_time}"
                )
        else:
            with open(output_file, 'a') as f:
                f.write(
                    f"{model_index}, {model.name}, {model.type}, {size_to_string(model.size/4)}, {model.size}, {model.task}, {model.url}, {model.downloads}, {enable_deepspeed}, {mean_time}"
                )
    else:
        ds_time = 0.0
        with open(output_file, 'r') as f:
            lastline = f.readlines()[-1]
            ds_time = float(lastline.split(",")[-1])
        if ds_time:
            with open(output_file, 'a') as f:
                print(f"{enable_deepspeed}, {mean_time}, {mean_time/ds_time}\n")
                f.write(f", {enable_deepspeed}, {mean_time}, {mean_time/ds_time}\n")

    # _kill_deployment(model)
