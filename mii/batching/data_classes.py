# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Iterator, Union
from typing_extensions import Self

import torch

from mii.constants import GenerationFinishReason
from mii.config import GenerateParamsConfig


@dataclass
class Response:
    """
    Response object returns from text-generation pipelines and persistent deployments.
    """

    generated_text: str
    """ The generated text. """

    prompt_length: int
    """ Number of tokens in the prompt. """

    generated_length: int
    """ Number of generated tokens. """

    finish_reason: GenerationFinishReason
    """ Reason for ending generation. One of :class:`mii.constants.GenerationFinishReason`. """
    @staticmethod
    def from_msg_dict(msg: Dict[str, Union[str, int]]) -> Self:
        return Response(**msg)

    def to_msg_dict(self) -> Dict[str, Union[str, int]]:
        return asdict(self)

    def __repr__(self) -> str:
        return self.generated_text

    def __str__(self) -> str:
        return self.generated_text


@dataclass
class RequestMsg:
    uid: int
    input_tokens: Union[torch.Tensor, List[int]]

    @property
    def is_flush_request(self):
        return self.input_tokens is None

    @staticmethod
    def from_msg_dict(msg: Dict[str, Any]) -> Self:
        input_tokens = msg["input_tokens"]
        if input_tokens is not None:
            input_tokens = torch.tensor(msg["input_tokens"],
                                        dtype=torch.int32,
                                        device=torch.device("cpu"))
        return RequestMsg(uid=msg["uid"], input_tokens=input_tokens)


@dataclass
class Request:
    tid: int
    uid: int
    input_tokens: torch.Tensor
    prompt_tokens: torch.Tensor
    seq_length: int
    last_in_prompt: bool
    post_processing: List[object]
    generate_params: GenerateParamsConfig

    _next_token: Union[None, torch.Tensor] = None
    _is_done: bool = False
    _generated_tokens: List[torch.Tensor] = field(default_factory=list)
    _finish_reason: GenerationFinishReason = GenerationFinishReason.NONE

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_tokens)

    @property
    def next_token(self) -> Union[None, torch.Tensor]:
        return self._next_token

    @property
    def ignore_eos(self) -> bool:
        return self.generate_params.ignore_eos

    @property
    def min_new_tokens(self) -> int:
        return self.generate_params.min_new_tokens

    @property
    def max_new_tokens(self) -> int:
        return self.generate_params.max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, max_new_tokens: int) -> None:
        self.generate_params.max_new_tokens = max_new_tokens

    @property
    def stream(self) -> bool:
        return self.generate_params.stream

    @property
    def return_full_text(self) -> bool:
        return self.generate_params.return_full_text

    @property
    def max_length(self) -> int:
        return self.generate_params.max_length

    @next_token.setter
    def next_token(self, next_token: Union[None, torch.Tensor]) -> None:
        self._next_token = next_token

    @property
    def is_done(self) -> bool:
        if self.ignore_eos:
            return False
        if self.seq_length < self.min_new_tokens:
            return False
        return self._is_done

    @is_done.setter
    def is_done(self, is_done: bool) -> None:
        self._is_done = is_done

    @property
    def generated_tokens(self) -> List[torch.Tensor]:
        return self._generated_tokens

    @property
    def finish_reason(self) -> GenerationFinishReason:
        return self._finish_reason

    @property
    def is_flush_request(self):
        return self.input_tokens is None

    @property
    def num_generated_tokens(self) -> int:
        # We return zero while we are processing decomposed prompts
        return self.seq_length - self.prompt_length + 1 if self.seq_length >= self.prompt_length else 0

    @property
    def stop_generation(self) -> bool:
        # Returns whether to stop generation for request
        if self.is_done:
            self._finish_reason = GenerationFinishReason.STOP
            return True
        if (self.seq_length >= self.max_length) or (self.num_generated_tokens >=
                                                    self.max_new_tokens):
            self._finish_reason = GenerationFinishReason.LENGTH
            return True
        return False

    def to_msg_dict(self) -> Dict[str, Any]:
        # Returns a minimal version of the request of purposes of broadcasting to all ranks
        input_tokens = self.input_tokens
        if input_tokens is not None:
            input_tokens = self.input_tokens.tolist()
        return {"uid": self.uid, "input_tokens": input_tokens}

    def accumulate_generated_token(self) -> None:
        # Append the latest token to the list of generated tokens
        if not self.is_done:
            self._generated_tokens.append(self.next_token)

    def clear_generated_token(self) -> None:
        self._generated_tokens.clear()

    def set_next_as_input(self) -> None:
        # Places the next token into the input token for next round of generation
        if self.next_token is not None:
            self.input_tokens = self.next_token.unsqueeze(0)
        self.last_in_prompt = True
        self.next_token = None
        self.is_done = False


class RequestBatch:
    def __init__(self, requests: List[Request] = None) -> None:
        if requests is None:
            requests = []
        self.requests = requests

    def __len__(self) -> int:
        return len(self.requests)

    def __contains__(self, r: Request) -> bool:
        return r in self.requests

    def __nonzero__(self) -> bool:
        if len(self.requests) != 0:
            return True
        return False

    def __iter__(self) -> Iterator[Request]:
        return iter(self.requests)

    def __repr__(self) -> str:
        return f"RequestBatch({self.requests})"

    @property
    def requests_to_run(self) -> Self:
        return RequestBatch([r for r in self.requests if not r.is_flush_request])

    @property
    def requests_to_flush(self) -> Self:
        return RequestBatch([r for r in self.requests if r.is_flush_request])

    @property
    def last_in_prompt(self) -> Self:
        return RequestBatch([r for r in self.requests if r.last_in_prompt])

    @property
    def completed(self) -> Self:
        return RequestBatch([r for r in self.requests if r.stop_generation])

    @property
    def uids(self) -> List[int]:
        return [r.uid for r in self.requests]

    @property
    def lengths(self) -> List[int]:
        return [len(r.input_tokens) for r in self.requests]

    @property
    def tokens(self) -> List[torch.Tensor]:
        return [r.input_tokens for r in self.requests]

    @property
    def next_tokens(self) -> List[torch.Tensor]:
        return [r.next_token for r in self.requests]

    @property
    def done_tokens(self) -> List[bool]:
        return [r.is_done for r in self.requests]

    @next_tokens.setter
    def next_tokens(self, next_tokens: torch.Tensor) -> None:
        assert len(next_tokens) == len(self.requests)
        for idx, r in enumerate(self.requests):
            r.next_token = next_tokens[idx]

    @done_tokens.setter
    def done_tokens(self, done_tokens: torch.Tensor) -> None:
        assert len(done_tokens) == len(self.requests)
        for idx, r in enumerate(self.requests):
            r.is_done = done_tokens[idx].item()

    def to_msg_dicts(self) -> List[Dict[str, Any]]:
        return [r.to_msg_dict() for r in self.requests]

    @staticmethod
    def from_msg_dicts(msg_dicts: List[Dict[str, Any]]) -> Self:
        return RequestBatch([RequestMsg.from_msg_dict(msg) for msg in msg_dicts])

    def prune(self, uids: List[int]) -> None:
        self.requests = [r for r in self.requests if r.uid not in uids]

    def append(self, r: Request) -> None:
        self.requests.append(r)

    def update_seq_length(self) -> None:
        for r in self.requests:
            r.seq_length += r.input_tokens.size(0)
