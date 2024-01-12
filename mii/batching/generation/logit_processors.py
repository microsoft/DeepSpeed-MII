# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import abc
from typing import List, Optional

import torch
import torch.nn.functional as F

FLOAT_PAD = -float("inf")


class BaseLogitProcessor(abc.ABC):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return self.forward(logits)

    @abc.abstractmethod
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        ...

    def get_key(self) -> str:
        return self.__class__.__name__


class TopKLogitProcessor(BaseLogitProcessor):
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
        logits[indices_to_remove] = FLOAT_PAD
        return logits

    def get_key(self) -> str:
        return super().get_key() + f"_top_k={self.top_k}"


class TopPLogitProcessor(BaseLogitProcessor):
    def __init__(self, top_p: float) -> None:
        assert 0.0 <= top_p <= 1.0
        self.top_p = top_p

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # convert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1,
                                                             sorted_indices,
                                                             sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, FLOAT_PAD)

    def get_key(self) -> str:
        return super().get_key() + f"_top_p={self.top_p}"


class TemperatureLogitProcessor(BaseLogitProcessor):
    def __init__(self, temperature: float) -> None:
        self.temperature = temperature
        assert self.temperature > 0.0

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def get_key(self) -> str:
        return super().get_key() + f"_temperature={self.temperature}"


class PipelineLogitProcessor(BaseLogitProcessor):
    def __init__(self, pipeline: List[BaseLogitProcessor]) -> None:
        assert all(isinstance(step, BaseLogitProcessor) for step in pipeline)
        self.pipeline = pipeline

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        for step in self.pipeline:
            logits = step(logits)
        return logits

    def get_key(self) -> str:
        return super().get_key(
        ) + f"_{'_'.join(step.get_key() for step in self.pipeline)}"


class NucleusSamplingLogitProcessor(BaseLogitProcessor):
    def __init__(self,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> None:
        assert top_k is not None or top_p is not None
        if top_k is None:
            self._processor = TopPLogitProcessor(top_p)
        elif top_p is None:
            self._processor = TopKLogitProcessor(top_k)
        else:
            self._processor = PipelineLogitProcessor(
                [TopKLogitProcessor(top_k),
                 TopPLogitProcessor(top_p)])

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self._processor(logits)

    def get_key(self) -> str:
        return super().get_key() + f"_{self._processor.get_key()}"
