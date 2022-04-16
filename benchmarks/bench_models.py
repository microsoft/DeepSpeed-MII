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
            return str(round(size / 10**3, 2)) + " k"
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

def _deploy_model(model, mii_configs=mii.constants.MII_CONFIGS_DEFAULT, enable_deepspeed=True):
    name = model.name
    type = model.type
    task = model.task
    mii.deploy(task,
            name,
            mii.DeploymentType.LOCAL,
            deployment_name=name + "_deployment",
            local_model_path=".cache/models/" + name,
            mii_configs=mii_configs,
            enable_deepspeed=True)

def _kill_deployment(model, mii_configs=mii.constants.MII_CONFIGS_DEFAULT):
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
    parser.add_argument("-f", "--model_file", type=str, default="sampled_models.json", help="Path to file containing model list")
    parser.add_argument("-o", "--output_file", type=str, default="bench_models_output.json", help="Path to file containing model list")
    parser.add_argument("-i", "--model_index", type=int, default=0, help="Index of model in the model_files to benchmark")
    parser.add_argument("-n", "--num_iters", type=int, help="number of iterations to run", default=10)
    parser.add_argument("--disable_deepspeed", action='store_true')
    args = parser.parse_args()

    model_file = args.model_file
    models = []
    if os.path.exists(model_file):
        with open(model_file) as f:
            data = f.read()
            models = Model.schema().loads(data, many=True)

    print(f"Populated {len(models)} models from file {model_file}")

    print(args)
    model_index = args.model_index
    model = models[model_index]
    num_iters = args.num_iters
    enable_deepspeed = not args.disable_deepspeed
    input = "DeepSpeed is the greatest"

    print(f" Benchmakring {model.type}, {model.task}, {size_to_string(model.size)} with enable_deepspeed={enable_deepspeed}")

    _deploy_model(model, enable_deepspeed=enable_deepspeed)

    time_takens = []
    for i in range(num_iters):
        if i < 1: continue # warmup
        generator = mii.mii_query_handle(model.name + "_deployment")
        result = generator.query({'query': input})
        time_takens.append(result.time_taken)

    mean_time = mean(time_takens)
    print(f"mean time_taken: {mean_time}")

    with open(args.output_file, 'a') as f:
        writer = csv.writer(f)
        row = [model.name, model.type, model.size, model.task, mean_time, enable_deepspeed, model.url]
        writer.writerow(row)

    _kill_deployment(model)
