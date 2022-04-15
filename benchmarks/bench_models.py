import subprocess
import sys
import os
import time

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

def _deploy_model(model, mii_configs=mii.constants.MII_CONFIGS_DEFAULT):
    name = model.name
    type = model.type
    task = model.task
    mii.deploy(task,
            name,
            mii.DeploymentType.LOCAL,
            deployment_name=name + "_deployment",
            local_model_path=".cache/models/" + name,
            mii_configs=mii_configs)

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
    read_file_path = "sampled_models.json"
    models = []

    if read_file_path and os.path.exists(read_file_path):
        print(f"Populating model list from file {read_file_path}...")
        with open(read_file_path) as f:
            data = f.read()
            models = Model.schema().loads(data, many=True)

    print(f"Populated {len(models)} models from file {read_file_path}")

    for model in models:
        print(f" Benchmakring {model.type}, {model.task}, {size_to_string(model.size)}")
        _deploy_model(model)
        time.sleep(10)
        _kill_deployment(model) # NEED A PROCESS REF TO KILL
        exit()
