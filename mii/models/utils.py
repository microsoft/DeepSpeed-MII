import os
from mii.utils import mii_cache_path


def supported_models_from_huggingface():
    return ['gpt2', "deepset/roberta-large-squad2"]


'''TODO make this more robust. If the pipeline has already been imported then
this might not work since the cache is set by the first import'''


def _download_hf_model_to_path(task, model_name, model_path):

    os.environ["TRANSFORMERS_CACHE"] = model_path
    from transformers import pipeline
    inference_pipeline = pipeline(task, model=model_name)


'''generic method that will allow downloading all models that we support.
Currently only supports HF models, but will be extended to support model checkpoints
from other sources'''


def download_model_and_get_path(task, model_name):

    model_path = os.path.join(mii_cache_path(), model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if model_name in supported_models_from_huggingface():
        _download_hf_model_to_path(task, model_name, model_path)
    else:
        assert False, "Only models from HF supported so far"

    return model_path
