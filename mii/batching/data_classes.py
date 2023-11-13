# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from dataclasses import dataclass, field
from typing import Dict, List, Iterator, Union
from typing_extensions import Self

import torch

from mii.constants import GenerationFinishReason


@dataclass
class Response:
    generated_text: str
    prompt_length: int
    generated_length: int
    finish_reason: GenerationFinishReason

    @staticmethod
    def from_msg(msg: Dict[str, Union[str, int]]) -> Self:
        return Response(
            generated_text=msg["generated_text"],
            prompt_length=msg["prompt_length"],
            generated_length=msg["generated_length"],
            finish_reason=GenerationFinishReason(msg["finish_reason"]),
        )

    def get_msg(self) -> Dict[str, Union[str, int]]:
        return {
            "generated_text": self.generated_text,
            "prompt_length": self.prompt_length,
            "generated_length": self.generated_length,
            "finish_reason": self.finish_reason.value
        }

    def __repr__(self) -> str:
        return self.generated_text

    def __str__(self) -> str:
        return self.generated_text


class ResponseBatch:
    def __init__(self, responses: List[Response]) -> None:
        self.responses = responses

    def __iter__(self) -> Iterator[Response]:
        return iter(self.responses)

    def __repr__(self) -> str:
        return "\n\n".join(str(r) for r in self.responses)

    @property
    def generated_texts(self) -> List[str]:
        return [r.generated_text for r in self.responses]

    @property
    def prompt_lengths(self) -> List[int]:
        return [r.prompt_length for r in self.responses]

    @property
    def generated_lengths(self) -> List[int]:
        return [r.generated_length for r in self.responses]

    @property
    def finish_reasons(self) -> List[GenerationFinishReason]:
        return [r.finish_reason for r in self.responses]

    def append(self, response: Response) -> None:
        self.responses.append(response)


@dataclass
class RaggedRequestMsg:
    uid: int
    input_tokens: Union[torch.Tensor, List[int]]

    @property
    def is_flush_request(self):
        return self.input_tokens is None

    @staticmethod
    def from_msg(msg: Dict[str, int]) -> Self:
        return RaggedRequestMsg(
            uid=msg["uid"],
            input_tokens=None
            if msg["input_tokens"] is None else torch.tensor(msg["input_tokens"],
                                                             dtype=torch.int32,
                                                             device=torch.device("cpu")),
        )


@dataclass
class RaggedRequest:
    tid: int
    uid: int
    input_tokens: torch.Tensor
    prompt_tokens: torch.Tensor
    seq_length: int
    max_length: int
    max_new_tokens: int
    min_new_tokens: int
    last_in_prompt: bool
    post_processing: List[object]
    stream: bool = False
    ignore_eos: bool = False
    return_full_text: bool = False

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
        if self.is_done:
            self._finish_reason = GenerationFinishReason.STOP
            return True
        if (self.seq_length >= self.max_length) or (self.num_generated_tokens >=
                                                    self.max_new_tokens):
            self._finish_reason = GenerationFinishReason.LENGTH
            return True
        return False

    def get_msg(self) -> RaggedRequestMsg:
        return RaggedRequestMsg(
            uid=self.uid,
            input_tokens=None
            if self.input_tokens is None else self.input_tokens.tolist(),
        )

    def accumulate_generated_token(self) -> None:
        if not self.is_done:
            self._generated_tokens.append(self.next_token)

    def clear_generated_token(self) -> None:
        self._generated_tokens.clear()

    def set_next_as_input(self) -> None:
        if self.next_token is not None:
            self.input_tokens = self.next_token.unsqueeze(0)
        self.last_in_prompt = True
        self.next_token = None
        self.is_done = False


class RaggedRequestBatch:
    def __init__(self, requests: List[RaggedRequest]) -> None:
        self.requests = requests

    def __len__(self) -> int:
        return len(self.requests)

    def __contains__(self, r: RaggedRequest) -> bool:
        return r in self.requests

    def __nonzero__(self) -> bool:
        if len(self.requests) != 0:
            return True
        return False

    def __iter__(self) -> Iterator[RaggedRequest]:
        return iter(self.requests)

    def __repr__(self) -> str:
        return f"RaggedRequestBatch({self.requests})"

    @property
    def requests_to_run(self) -> Self:
        return RaggedRequestBatch([r for r in self.requests if not r.is_flush_request])

    @property
    def requests_to_flush(self) -> Self:
        return RaggedRequestBatch([r for r in self.requests if r.is_flush_request])

    @property
    def last_in_prompt(self) -> Self:
        return RaggedRequestBatch([r for r in self.requests if r.last_in_prompt])

    @property
    def completed(self) -> Self:
        return RaggedRequestBatch([r for r in self.requests if r.stop_generation])

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
    def done_tokens(self) -> List[torch.Tensor]:
        return [r.is_done for r in self.requests]

    @next_tokens.setter
    def next_tokens(self, next_tokens: List[torch.Tensor]) -> None:
        assert len(next_tokens) == len(self.requests)
        for idx, r in enumerate(self.requests):
            r.next_token = next_tokens[idx]

    @done_tokens.setter
    def done_tokens(self, done_tokens: List[torch.Tensor]) -> None:
        assert len(done_tokens) == len(self.requests)
        for idx, r in enumerate(self.requests):
            r.is_done = done_tokens[idx]

    def prune(self, uids: List[int]) -> None:
        self.requests = [r for r in self.requests if r.uid not in uids]

    def append(self, r: RaggedRequest) -> None:
        self.requests.append(r)

    def update_seq_length(self) -> None:
        for r in self.requests:
            r.seq_length += r.input_tokens.size(0)