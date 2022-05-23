from cgitb import enable
from re import I
import subprocess
import sys
import os
import time
import argparse
import csv

from statistics import mean
from get_hf_models import Model
import mii

# MODEL_TYPE = "roberta_81.53M_357.02M_34"
# MODEL_TYPE="gpt2_629.14K_1.61G_12"
# MODEL_TYPE="bert_61.44K_335.54M_40"
MODEL_TYPE = "gpt_neo_83.36M_2.67G_4"
# MODEL_TYPE="gptj_6.04G_6.04G_1"

DATA_TYPE = "fp16"
MODEL_FILE = "sampled_models_$MODEL_TYPE.json"
OUTPUT_FILE = "output_${MODEL_TYPE}_${DATA_TYPE}.csv"

model_file = "sampled_models_" + MODEL_TYPE + ".json"
output_file = "output_" + MODEL_TYPE + "_" + DATA_TYPE + ".csv"


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


assert os.path.exists(model_file), f"file {model_file} does not exist"
assert os.path.exists(output_file), f"file {output_file} does not exist"

with open(model_file) as f:
    data = f.read()
    models = Model.schema().loads(data, many=True)

with open(output_file, 'r') as f:
    lines = f.readlines()

existing_ids = []
for l in lines:
    id = l.split(",")[0]
    if id != "":
        id = int(id)
        existing_ids.append(id)
print(f"existing_ids = {existing_ids}")

missing_out = "missing_" + MODEL_TYPE + "_" + DATA_TYPE + ".csv"
id = 0
for m in models:
    if id not in existing_ids:
        print(f"{id}, {m.name} is missing")
        with open(missing_out, 'a') as f:
            f.write(
                f"{id}, {m.name}, {m.type}, {size_to_string(m.size)}, {m.size}, {m.task}, {m.url}, {m.downloads}\n"
            )
    id += 1
