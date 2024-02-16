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
from transformers import AutoConfig, AutoModel, GenerationConfig
from typing import Iterable, Tuple


class RandomWeightsCheckpointEngine(CheckpointEngineBase):

    # When using AutoModel.from_config() to load the model, the layer names are
    # often missing a prefix. We default to adding "model." as the prefix, but
    # others can be specified here.
    layer_prefix_map = {"falcon": "transformer."}

    # When using AutoModel.from_config() to load the model, the lm_head layer is
    # not generated. We default to populating this with the
    # "embed_tokens.weight" layer, but others can be specified here.
    lm_head_layer_map = {"falcon": "word_embeddings.weight"}

    def __init__(self, model_name_or_path: str, auth_token: str = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_seq_length = self.model_config.max_position_embeddings
        else:
            try:
                generation_config = GenerationConfig.from_pretrained(
                    self.model_name_or_path)
                self.model_config.max_seq_length = generation_config.max_length
            except OSError:
                self.model_config.max_seq_length = 2048

    def _get_layer_prefix(self) -> str:
        for model_type, prefix in self.layer_prefix_map.items():
            if model_type in self.model_name_or_path.lower():
                return prefix
        return "model."

    def _get_lm_head_layer(self) -> str:
        for model_type, layer in self.lm_head_layer_map.items():
            if model_type in self.model_name_or_path.lower():
                return layer
        return "embed_tokens.weight"

    def parameters(self) -> Iterable[Tuple[str, torch.Tensor]]:
        layer_prefix = self._get_layer_prefix()
        lm_head_layer = self._get_lm_head_layer()

        # Load with meta device is faster
        with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
            model = AutoModel.from_config(self.model_config)

        for param_name, param in model.state_dict().items():
            yield layer_prefix + param_name, torch.zeros(param.shape)
            if param_name == lm_head_layer:
                yield "lm_head.weight", torch.zeros(param.shape)


@pytest.fixture(scope="module", autouse=True)
def inject_checkpoint_engine():
    # Inject the random weihts checkpoint engine
    deepspeed.inference.v2.engine_factory.HuggingFaceCheckpointEngine = (
        RandomWeightsCheckpointEngine)
    yield None
    # Restore the original checkpoint engine
    deepspeed.inference.v2.engine_factory.HuggingFaceCheckpointEngine = (
        HuggingFaceCheckpointEngine)


@pytest.mark.parametrize(
    "model_name",
    [
        "tiiuae/falcon-7b",
        "NousResearch/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
        "cloudyu/Mixtral_11Bx2_MoE_19B",
        "facebook/opt-125m",
    ],
    ids=["falcon",
         "llama",
         "mistral",
         "mixtral",
         "opt"],
)
def test_model(pipeline, query):
    outputs = pipeline(query, max_new_tokens=16)
    assert outputs[0], "output is empty"


@pytest.mark.parametrize("local_model", [True])
def test_local_model_dir(pipeline):
    assert pipeline
