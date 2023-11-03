# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from typing import Optional, Any, Dict

from mii.batching import MIIPipeline, MIIAsyncPipeline
from mii.config import ModelConfig
from mii.models import load_model
from mii.tokenizers import load_tokenizer


def pipeline(model_name_or_path: str = "",
             model_config: Optional[Dict[str,
                                         Any]] = None,
             **kwargs) -> MIIPipeline:
    if model_config is None:
        model_config = {}
    if model_name_or_path:
        if "model_name_or_path" in model_config:
            assert model_config.get("model_name_or_path") == model_name_or_path, "model_name_or_path in model_config must match model_name_or_path"
        model_config["model_name_or_path"] = model_name_or_path
    for key, val in kwargs.items():
        if key in ModelConfig.__dict__["__fields__"]:
            if key in model_config:
                assert model_config.get(key) == val, f"{key} in model_config must match {key}"
            model_config[key] = val
        else:
            raise ValueError(f"Invalid keyword argument {key}")
    model_config = ModelConfig(**model_config)

    inference_engine = load_model(model_config)
    tokenizer = load_tokenizer(model_config)
    inference_pipeline = MIIPipeline(inference_engine=inference_engine,
                                     tokenizer=tokenizer,
                                     model_config=model_config)
    return inference_pipeline


def async_pipeline(model_config: ModelConfig) -> MIIAsyncPipeline:
    inference_engine = load_model(model_config)
    tokenizer = load_tokenizer(model_config)
    inference_pipeline = MIIAsyncPipeline(inference_engine=inference_engine,
                                          tokenizer=tokenizer,
                                          model_config=model_config)
    return inference_pipeline
