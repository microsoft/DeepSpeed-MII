# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.inference import build_hf_engine, InferenceEngineV2

from mii.config import ModelConfig
from mii.constants import ModelProvider
from mii.utils import init_distributed


def load_model(model_config: ModelConfig) -> InferenceEngineV2:
    init_distributed(model_config)
    provider = model_config.provider
    if provider == ModelProvider.HUGGING_FACE:
        inference_engine = build_hf_engine(
            path=model_config.model_name_or_path,
            engine_config=model_config.inference_engine_config)
    else:
        raise ValueError(f"Unknown model provider {provider}")

    return inference_engine
