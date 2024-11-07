# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch

from .utils import attempt_load
from mii.config import ModelConfig


def diffusers_provider(model_config: ModelConfig):
    from diffusers import DiffusionPipeline

    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    kwargs = model_config.pipeline_kwargs
    if model_config.dtype == torch.half:
        kwargs["torch_dtype"] = torch.float16
        kwargs["revision"] = "fp16"

    pipeline = attempt_load(DiffusionPipeline.from_pretrained,
                            model_config.model,
                            model_config.model_path,
                            kwargs=kwargs)
    pipeline = pipeline.to(f"cuda:{local_rank}")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline
