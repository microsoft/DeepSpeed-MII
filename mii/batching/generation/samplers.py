# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import abc
from typing import Tuple

import torch
from torch.distributions import Categorical


class BaseGenerationSampler(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.LongTensor,
               torch.Tensor]:
        """
        Given the logits, return the next token to add to the sequence, as well
        as the log probability of the token

        Args:
            logits (torch.Tensor):
                The logits from the model. Shape: (batch_size, vocab_size)

        Returns:
            Tuple[torch.LongTensor, torch.Tensor]:
                The next token to add to the sequence, and the log probability
                of the token. Shape: (batch_size,) and (batch_size,)
        """
        ...

    def get_key(self) -> str:
        return self.__class__.__name__


class LogitsSampler(BaseGenerationSampler):
    def __call__(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.LongTensor,
               torch.Tensor]:
        logits = logits.float()
        sampler = Categorical(logits=logits)
        next_tokens = sampler.sample()
        #logprobs = sampler.log_prob(next_tokens)
        return next_tokens  #, logprobs


class GreedySampler(BaseGenerationSampler):
    def __call__(self, logits: torch.Tensor) -> Tuple[torch.LongTensor, torch.Tensor]:
        logits = logits.float()
        #sampler = Categorical(logits=logits)
        next_tokens = logits.argmax(dim=-1)
        #logprobs = sampler.log_prob(next_tokens)
        return next_tokens  #, logprobs
