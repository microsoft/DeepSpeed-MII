# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch


def diffusers_provider(deployment_config):
    from diffusers import DiffusionPipeline

    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    kwargs = {}
    if deployment_config.dtype == torch.half:
        kwargs["torch_dtype"] = torch.float16
        kwargs["revision"] = "fp16"

    pipeline = DiffusionPipeline.from_pretrained(
        deployment_config.model, use_auth_token=deployment_config.hf_auth_token, **kwargs
    )
    pipeline = pipeline.to(f"cuda:{local_rank}")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline
