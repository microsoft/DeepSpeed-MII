# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import deepspeed
import torch
from deepspeed.inference.v2.checkpoint import (
    CheckpointEngineBase,
    HuggingFaceCheckpointEngine,
)
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from typing import Iterable, Tuple


class ZeroWeightsCheckpointEngine(CheckpointEngineBase):
    """ Generates weight with all zeros for a given model for testing purposes. """
    def __init__(self, model_name_or_path: str, auth_token: str = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.model_config = AutoConfig.from_pretrained(self.model_name_or_path,
                                                       trust_remote_code=True)
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_seq_length = self.model_config.max_position_embeddings
        else:
            try:
                generation_config = GenerationConfig.from_pretrained(
                    self.model_name_or_path)
                self.model_config.max_seq_length = generation_config.max_length
            except OSError:
                self.model_config.max_seq_length = 2048

    def parameters(self) -> Iterable[Tuple[str, torch.Tensor]]:
        # Load with meta device is faster
        with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
            model = AutoModelForCausalLM.from_config(self.model_config,
                                                     trust_remote_code=True)

        for param_name, param in model.state_dict().items():
            yield param_name, torch.zeros(param.shape)


@pytest.fixture(scope="module", autouse=True)
def inject_checkpoint_engine():
    # Inject the random weihts checkpoint engine
    deepspeed.inference.v2.engine_factory.HuggingFaceCheckpointEngine = (
        ZeroWeightsCheckpointEngine)
    yield None
    # Restore the original checkpoint engine
    deepspeed.inference.v2.engine_factory.HuggingFaceCheckpointEngine = (
        HuggingFaceCheckpointEngine)


@pytest.mark.parametrize(
    "model_name",
    [
        "tiiuae/falcon-7b",
        "huggyllama/llama-7b",
        "NousResearch/Llama-2-7b-hf",
        "NousResearch/Hermes-2-Pro-Mistral-7B",
        "cloudyu/Mixtral_11Bx2_MoE_19B",
        "facebook/opt-125m",
        "microsoft/phi-2",
        "Qwen/Qwen-7B-Chat",
        "Qwen/Qwen1.5-0.5B",
    ],
    ids=[
        "falcon",
        "llama",
        "llama-2",
        "mistral",
        "mixtral",
        "opt",
        "phi-2",
        "qwen",
        "qwen-2"
    ],
)
def test_model(pipeline, query):
    outputs = pipeline(query, max_new_tokens=16)
    assert outputs[0], "output is empty"


@pytest.mark.parametrize("local_model", [True])
def test_local_model_dir(pipeline):
    assert pipeline
