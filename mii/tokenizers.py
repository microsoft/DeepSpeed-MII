# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import torch
from transformers import AutoTokenizer

from mii.constants import ModelProvider

if TYPE_CHECKING:
    from mii.config import ModelConfig


class MIITokenizerWrapper(ABC):
    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        ...

    @abstractmethod
    def encode(self, input: str) -> torch.Tensor:
        ...

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        ...


class HFTokenizer(MIITokenizerWrapper):
    def __init__(self, tokenizer: Union[str, object]) -> None:
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            tokenizer.pad_token = tokenizer.eos_token
        super().__init__(tokenizer)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def encode(self, input: str) -> torch.Tensor:
        return self.tokenizer.encode(input, return_tensors="pt").flatten()

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens)


def load_tokenizer(model_config: "ModelConfig") -> MIITokenizerWrapper:
    provider = model_config.provider
    if provider == ModelProvider.HUGGING_FACE:
        tokenizer = HFTokenizer(model_config.tokenizer)
    else:
        raise ValueError(f"Unknown model provider {provider}")

    return tokenizer
