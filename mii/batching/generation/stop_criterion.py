# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import abc
from typing import List, Union

import torch

# from megatron import get_tokenizer
# from megatron.tokenizer.tokenizer import AbstractTokenizer


class BaseGenerationStopCriterion(abc.ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, tokens: torch.LongTensor) -> torch.BoolTensor:
        return self.forward(tokens)

    @abc.abstractmethod
    def forward(self, tokens: torch.LongTensor) -> torch.BoolTensor:
        ...

    def get_key(self) -> str:
        return self.__class__.__name__


class TokenStopCriterion(BaseGenerationStopCriterion):
    def __init__(self, token: Union[str, int], tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        if isinstance(token, str):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
        else:
            token_id = token
        self.stop_token_id = token_id

    def forward(self, tokens: torch.LongTensor) -> torch.BoolTensor:
        retval = torch.zeros_like(tokens, dtype=torch.bool)
        retval |= tokens == self.stop_token_id
        return retval

    def get_key(self) -> str:
        return self.__class__.__name__ + f"_token_id={self.stop_token_id}"


class EosGenerationStopCriterion(BaseGenerationStopCriterion):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)
        if hasattr(self.tokenizer, "eod"):
            self.eos_id = self.tokenizer.eod
        elif hasattr(self.tokenizer, "eos_token_id"):
            self.eos_id = self.tokenizer.eos_token_id
        elif hasattr(self.tokenizer, "eos_token"):
            self.eos_id = self.tokenizer.eos_token
        else:
            raise ValueError(
                "Tokenizer must have either an `eod` or `eos_token` attribute.")

    def forward(self, tokens: torch.LongTensor) -> torch.BoolTensor:
        return tokens == self.eos_id


class NewLineDelimitedStopCriterion(BaseGenerationStopCriterion):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.stop_token_ids = list(
            set([self.tokenizer.tokenize(x)[0] for x in ["\n",
                                                         "\r\n",
                                                         "\n\n",
                                                         ".\n\n"]]))

    def forward(self, tokens: torch.LongTensor) -> torch.BoolTensor:
        retval = torch.zeros_like(tokens, dtype=torch.bool)
        for stop_token_id in self.stop_token_ids:
            retval |= tokens == stop_token_id
        return retval


class PipelinedCriterion(BaseGenerationStopCriterion):
    def __init__(
        self,
        criteria: List[BaseGenerationStopCriterion],
        tokenizer,
    ):
        super().__init__(tokenizer=tokenizer)
        self.criteria = criteria

    def forward(self, tokens: torch.LongTensor) -> torch.BoolTensor:
        retval = torch.zeros_like(tokens, dtype=torch.bool)
        for criterion in self.criteria:
            retval |= criterion(tokens)
        return retval

    def get_key(self) -> str:
        return super().get_key(
        ) + f"_{'_'.join(criterion.get_key() for criterion in self.criteria)}"
