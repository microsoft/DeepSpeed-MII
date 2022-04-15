from cmath import inf
import requests
import re
import json
import os

from transformers import CONFIG_MAPPING
from huggingface_hub import HfApi
from dataclasses_json import dataclass_json
from dataclasses import dataclass

api = HfApi()


@dataclass_json
@dataclass(frozen=True,repr=True,eq=True)
class Model:
    name: str
    type: str
    url: str
    size: int = 0
    downloads: int = 0
    def __lt__(self, other):
        return self.size > other.size or (self.size == other.size and self.downloads > other.downloads)

def _get_models_by_type_and_task(model_type, task=None):
    models = api.list_models(filter=model_type)
    # sort by number of downloads
    ordered = sorted(models,
                     reverse=True,
                     key=lambda t: t.downloads
                     if hasattr(t, "downloads") else 0)
    return [ m for m in ordered if m.pipeline_tag and m.pipeline_tag == task] if task else [ m for m in ordered]

def _get_model_size(model_name):
    url = f"https://huggingface.co/{model_name}/blob/main/pytorch_model.bin"
    response = requests.get(url)
    if response.status_code != 200: # only check pytorch models
        return -1
    data = response.text
    regex = r"<strong>Size of remote file:<\/strong>\s*([\d\.]+)\s+(\w+)<\/li>"
    found = re.findall(regex, data, re.MULTILINE)
    size, unit = found[0]
    size = float(size)
    if unit == "KB":
        size *= 1024
    elif unit == "MB":
        size *= 1024 * 1024
    elif unit == "GB":
        size *= 1024 * 1024 * 1024
    elif unit == "TB":
        size *= 1024 * 1024 * 1024 * 1024
    return size

def _populate_model_list(model_type, tasks, total_count=None, write_file_path="models.json", read_file_path=None):
    models = []
    cnt = 0

    # populate models dict from a json file
    if read_file_path and os.path.exists(read_file_path):
        print(f"Populating model list from file {read_file_path}...")
        with open(read_file_path) as f:
            data = f.read()
            models = Model.schema().loads(data, many=True)
    if models:
        print(f"Populated {len(models)} models from file {read_file_path}")
        return models

    print(f"Populating model list from hf hub...")
    # populate models dict from hf-hub
    for mt in model_types:
        ms = _get_models_by_type_and_task(mt)
        for m in ms:
            try:
                if hasattr(m, "pipeline_tag") and m.pipeline_tag in tasks:
                    name = m.modelId
                    type = mt
                    url = f"https://huggingface.co/{name}"
                    size = _get_model_size(name)
                    if size == -1:
                        continue
                    downloads = m.downloads if hasattr(m, "downloads") else 0
                    mm = Model(name, type, url, size, downloads)
                    models.append(mm)
                    cnt += 1
                    if total_count and cnt == total_count:
                        print(f"Populated {len(models)} models from hf hub")
                        _write_model_list_to_file(models, write_file_path)
                        return models
            except:
                print("error cannot parse ", f"https://huggingface.co/{m.modelId}")

    print(f"Populated {len(models)} models from hf hub")
    if write_file_path:
        _write_model_list_to_file(models, write_file_path)
    return models

def _write_model_list_to_file(models, file_path):
    models_json = Model.schema().dumps(models, many=True)
    with open(file_path, 'w') as f:
        f.write(models_json)
    print(f"Wrote {len(models)} models to file {file_path}")

def _sample_models(models, total=5, bin_top_k=1, model_type=None, task=None):
    sampled_models = []
    models.sort()
    filtered_models = []
    min_size = float('inf')
    max_size = 0.0
    for m in models:
        if model_type:
            if m.type != model_type:
                continue
        if task:
            if m.pipeline_tag != task:
                continue
        filtered_models.append(m)
        if m.size > max_size:
            max_size = m.size
        if m.size < min_size:
            min_size = m.size

    step = (len(filtered_models) + total-1 ) // total

    for i in range(0, len(filtered_models), step):
        left = i
        right = min(i+step, i+bin_top_k)
        ordered = sorted(models[left:right], reverse=True,
                     key=lambda t: t.downloads
                     if hasattr(t, "downloads") else 0)
        sampled_models.append(ordered[0])
    return sampled_models, min_size, max_size


if __name__ == "__main__":
    model_types = ["roberta", "gpt2", "bert"]
    tasks = ["question-answering", "text-generation", "fill-mask", "token-classification", "conversational", "text-classification" ]

    # model_list = _populate_model_list(model_types, tasks)
    model_list = _populate_model_list(model_types, tasks, total_count = 40, read_file_path = "models.json")

    sampled_models, min_size, max_size = _sample_models(model_list, 10)
    _write_model_list_to_file(sampled_models, "sampled_models.json")

    print(f"Sampled {len(sampled_models)} models from size between {min_size} and {max_size}")

