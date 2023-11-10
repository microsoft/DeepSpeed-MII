# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import copy
import queue
import os
import random
import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict, field
from functools import cached_property
from typing import Dict, Tuple, List, Any, Iterator, Union, DefaultDict
from typing_extensions import Self

import torch
import ujson
import zmq
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.timer import SynchronizedWallClockTimer

from mii.batching.constants import (MAX_LENGTH_KWARG,
                                    MAX_NEW_TOKENS_KWARG,
                                    MIN_NEW_TOKENS_KWARG,
                                    STREAM_KWARG,
                                    IGNORE_EOS_KWARG,
                                    TOP_P_KWARG,
                                    TOP_K_KWARG,
                                    TEMPERATURE_KWARG,
                                    RETURN_FULL_TEXT_KWARG,
                                    DO_SAMPLE_KWARG,
                                    STOP_KWARG,
                                    MIN_NEW_TOKENS_DEFAULT,
                                    STREAM_DEFAULT,
                                    IGNORE_EOS_DEFAULT,
                                    TOP_P_DEFAULT,
                                    RETURN_FULL_TEXT_DEFAULT,
                                    DO_SAMPLE_DEFAULT,
                                    TOP_K_NAME,
                                    TOP_P_NAME,
                                    TEMP_NAME,
                                    SAMPLER_NAME,
                                    STOP_NAME)
from mii.batching.generation.logit_processors import TopPLogitProcessor, TopKLogitProcessor, TemperatureLogitProcessor
from mii.batching.generation.samplers import LogitsSampler, GreedySampler
from mii.batching.generation.stop_criterion import EosGenerationStopCriterion, TokenStopCriterion
from mii.batching.postprocess import (
    run_batch_logit_processing,
    run_batch_sampler,
    run_batch_stop_criterion,
)
from mii.batching.utils import sync_debug, profiler
from mii.constants import GenerationFinishReason, ZMQ_RECV_TIMEOUT
from mii.logging import logger


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


class RaggedBatchBase:
    def __init__(self, inference_engine, tokenizer, model_config):
        self.inference_engine = inference_engine
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.model_config = model_config
        self.zmq_port = model_config.zmq_port_number
        if model_config.max_length is not None:
            self.max_length = model_config.max_length
        else:
            self.max_length = inference_engine._policy._checkpoint_engine.model_config.max_seq_length
        self.sync_debug = model_config.sync_debug
        self.profile_model_time = model_config.profile_model_time

        self.request_queue: queue.Queue = queue.Queue()
        self.result_queues: Dict[int, queue.Queue] = {}
        self.scheduled_requests: RaggedRequestBatch = RaggedRequestBatch([])
        self.buffer = deque()
        self.scheduled_length = 0
        self.scheduled_seq_num = 0
        self.scheduled_req_blocks = 0

        # TODO: we will need to prune self._post_processors for long running deployments
        self._post_processors = {}
        self.logit_processor = run_batch_logit_processing
        self.sampler = run_batch_sampler
        self.stop_criterion = run_batch_stop_criterion

        self._timers: SynchronizedWallClockTimer = SynchronizedWallClockTimer()
        self._profiled_times: DefaultDict[str, List[int]] = defaultdict(list)
        self._iters: int = 0
        self._num_generated_tokens: int = 0

        context = zmq.Context()
        torch.cuda.synchronize()
        if self.is_rank_0:
            self.socket = context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.zmq_port}")
            time.sleep(1)  # Give the subscriber a change to connect
        else:
            self.socket = context.socket(zmq.SUB)
            self.socket.connect(f"tcp://localhost:{self.zmq_port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT)

    @cached_property
    def local_rank(self) -> int:
        return get_accelerator().current_device()

    @property
    def is_rank_0(self) -> bool:
        return self.local_rank == 0

    @profiler
    def generate(self) -> None:
        # 1. Get a batch of requests, broadcast to all ranks
        scheduled_requests = self._bcast_requests()

        # 2. Flush for uids that are finished generating
        self.flush(scheduled_requests.requests_to_flush.uids)

        # 3. Put new tokens into inference engine
        if scheduled_requests.requests_to_run:
            next_token_logits = self.put(
                scheduled_requests.requests_to_run.uids,
                scheduled_requests.requests_to_run.tokens,
            )

        # short circuit if not rank 0, only rank 0 does scheduling and postprocessing of logits
        if not self.is_rank_0:
            return

        # 4. Launch logit processing and token generation
        running_requests = scheduled_requests.requests_to_run
        running_requests.update_seq_length()
        if running_requests:
            next_tokens, done_tokens = self._process_logits(
                next_token_logits, running_requests
            )
            running_requests.next_tokens = next_tokens
            running_requests.done_tokens = done_tokens

        # 5. Schedule requests while we wait for the forward pass to finish
        self._reset_scheduler_bookkeeping()

        # 6. Accumulate generated tokens, check completion, and generate output
        for r in running_requests.last_in_prompt:
            r.accumulate_generated_token()
            self._num_generated_tokens += 1
            if r.stop_generation or r.stream:
                self._generate_output(r)
            if not r.stop_generation:
                r.set_next_as_input()
                self.request_queue.put(r)

        # 7. Update scheduled requests
        self.scheduled_requests.prune(running_requests.completed.uids)
        self.schedule_requests()

        if self.profile_model_time:
            self._print_profiled_times()

    def _print_profiled_times(self) -> None:
        self._iters += 1
        if not (self._iters % 100 == 0):
            return
        for event, times in self._profiled_times.items():
            mean_time = sum(times) / len(times)
            log_msg = f"{event}: {mean_time}"
            if event == "generate":
                log_msg += f" ({self._num_generated_tokens / sum(times)} tokens/ms)"
            logger.info(log_msg)
        self._profiled_times.clear()
        self._num_generated_tokens = 0

    @sync_debug
    def _bcast_requests(self, force=False) -> RaggedRequestBatch:
        if self.is_rank_0:
            if not self.scheduled_requests and not force:
                return self.scheduled_requests
            # Rank 0 gets batch of requests and broadcasts to other ranks
            data_dicts = [asdict(r.get_msg()) for r in self.scheduled_requests]
            json_data = ujson.dumps(data_dicts)
            self.socket.send_string(json_data)
        else:
            try:
                json_data = self.socket.recv_string()
                data_dicts = ujson.loads(json_data)
                self.scheduled_requests = RaggedRequestBatch(
                    [RaggedRequestMsg.from_msg(msg) for msg in data_dicts])
            except zmq.Again:
                self.scheduled_requests = RaggedRequestBatch([])

        return self.scheduled_requests

    def _reset_scheduler_bookkeeping(self) -> None:
        self.scheduled_requests = RaggedRequestBatch([])
        self.scheduled_length = 0
        self.scheduled_seq_num = 0
        self.scheduled_req_blocks = 0

    @sync_debug
    def _process_logits(
            self,
            next_token_logits: torch.Tensor,
            running_requests: RaggedRequestBatch) -> Tuple[torch.Tensor,
                                                           torch.Tensor]:
        next_token_logits = next_token_logits[:, :self.vocab_size]
        next_token_logits = self.logit_processor(next_token_logits,
                                                 running_requests,
                                                 self._post_processors)
        next_tokens = self.sampler(next_token_logits,
                                   running_requests,
                                   self._post_processors)
        done_tokens = self.stop_criterion(next_tokens,
                                          running_requests,
                                          self._post_processors)
        next_tokens = next_tokens.to(torch.device("cpu"), non_blocking=False)
        return next_tokens, done_tokens

    @sync_debug
    def _generate_output(self, r: RaggedRequest) -> bool:
        outputs = []
        if r.stream:
            outputs.append((
                r.uid,
                [r.next_token],
                r.prompt_length,
                r.num_generated_tokens,
                GenerationFinishReason.NONE,
            ))
        if r.finish_reason != GenerationFinishReason.NONE:
            if r.stream or not r.generated_tokens:
                output_tokens = []
            else:
                output_tokens = torch.cat([t.unsqueeze(0) for t in r.generated_tokens],
                                          dim=0)
                if r.return_full_text:
                    # Avoid returning bos token, refactor this later
                    output_tokens = torch.cat((r.prompt_tokens[1:], output_tokens))
            outputs.append((
                r.uid,
                output_tokens,
                r.prompt_length,
                r.num_generated_tokens,
                r.finish_reason,
            ))
        for output in outputs:
            self.result_queues[r.tid].put_nowait(output)

    def _do_schedule_requests(self, requests: List[RaggedRequest]) -> None:

        free_blocks = self.inference_engine._state_manager.free_blocks
        conf_manager = self.inference_engine._config.state_manager
        for r in requests:
            if r.max_length <= r.seq_length:
                continue

            # Make sure that the engine has enough capacity to process the batch
            if len(self.scheduled_requests) > conf_manager.max_ragged_sequence_count:
                break

            max_batch_size = conf_manager.max_ragged_batch_size - self.scheduled_length
            if max_batch_size <= 0:
                break

            max_blocks = free_blocks - self.scheduled_req_blocks

            # Check capacity to mitigate the deadlock risk
            # We don't schedule requests when we find that a prompt is too long to fit to the KV cache
            if len(r.input_tokens) > 1:
                req_tokens, _ = self.inference_engine.query(r.uid, len(r.input_tokens), max_blocks)
                if req_tokens < len(r.input_tokens):
                    break

            req_tokens = min(len(r.input_tokens), max_batch_size)
            req_tokens, req_blocks = self.inference_engine.query(r.uid, req_tokens, max_blocks)

            if req_tokens <= 0:
                continue

            # Decompose the prompt to fit to the max ragged batch size
            decomposed = req_tokens < len(r.input_tokens)
            remaining_tokens = r.input_tokens[req_tokens:]
            r.input_tokens = r.input_tokens[:req_tokens]
            r.last_in_prompt = not decomposed

            # Schedule the request
            self.scheduled_requests.append(r)

            self.scheduled_req_blocks += req_blocks
            self.scheduled_length += req_tokens

            if decomposed:
                req_remaining = copy.copy(r)
                req_remaining.input_tokens = remaining_tokens
                req_remaining.seq_length = r.seq_length + req_tokens
                req_remaining.last_in_prompt = True

                self.buffer.appendleft(req_remaining)

    def schedule_requests(self) -> None:
        while not self.request_queue.empty():
            r = self.request_queue.get_nowait()
            self.buffer.append(r)

        # Run next token generation first
        next_token_gen_reqs = []
        prompt_reqs = []

        for r in self.buffer:
            if r.is_flush_request:
                self.scheduled_requests.append(r)
            else:
                if len(r.input_tokens) == 1:
                    next_token_gen_reqs.append(r)
                else:
                    prompt_reqs.append(r)

        # We want to process next token generation first
        self._do_schedule_requests(next_token_gen_reqs)
        self._do_schedule_requests(prompt_reqs)

        if len(self.buffer) > 0 and len(self.scheduled_requests) == 0:
            print(
                "Deadlock detected. Resetting KV cache and recomputing requests. Consider limiting number of concurrent requests or decreasing max lengths of prompts/generations."
            )
            self.scheduled_requests = RaggedRequestBatch([])
            self.reset_request_status()
        else:
            scheduled_requests_ids = set(id(r) for r in self.scheduled_requests)
            self.buffer = deque(
                [r for r in self.buffer if id(r) not in scheduled_requests_ids])

    def _queue_flush_request(self, uid: int) -> None:
        self.request_queue.put_nowait(
            RaggedRequest(
                tid=None,
                uid=uid,
                input_tokens=None,
                prompt_tokens=None,
                seq_length=None,
                max_length=None,
                max_new_tokens=None,
                min_new_tokens=None,
                last_in_prompt=None,
                post_processing=None,
                stream=None,
            ))

    def reset_request_status(self):
        for r in self.buffer:
            if r.seq_length > 0:
                self._queue_flush_request(r.uid)

        new_buffer = deque()
        for r in self.buffer:
            new_req = copy.copy(r)
            new_req.prompt_tokens = new_req.input_tokens = torch.concat(
                [r.prompt_tokens] + [t.unsqueeze(0) for t in r.generated_tokens])
            new_req.seq_length = 0
            new_req.max_new_tokens = r.max_new_tokens - len(r.generated_tokens)
            new_req.clear_generated_token()
            new_buffer.append(new_req)

        self.buffer = new_buffer

    def make_request(self,
                     tid: int,
                     uid: int,
                     input_tokens: torch.Tensor,
                     kwargs: Dict) -> RaggedRequest:
        prompt_length = len(input_tokens)
        max_length = kwargs.pop(MAX_LENGTH_KWARG, self.max_length)
        assert max_length > prompt_length, f"prompt length must be less than {MAX_LENGTH_KWARG}"
        max_new_tokens = kwargs.pop(MAX_NEW_TOKENS_KWARG, max_length - prompt_length)
        min_new_tokens = kwargs.pop(MIN_NEW_TOKENS_KWARG, MIN_NEW_TOKENS_DEFAULT)
        stream = kwargs.pop(STREAM_KWARG, STREAM_DEFAULT)
        ignore_eos = kwargs.pop(IGNORE_EOS_KWARG, IGNORE_EOS_DEFAULT)
        return_full_text = kwargs.pop(RETURN_FULL_TEXT_KWARG, RETURN_FULL_TEXT_DEFAULT)

        post_processing = []

        top_p = kwargs.pop(TOP_P_KWARG, TOP_P_DEFAULT)
        top_p_name = "_".join((TOP_P_NAME, str(top_p)))
        if top_p_name not in self._post_processors:
            self._post_processors[top_p_name] = TopPLogitProcessor(top_p=top_p)
        post_processing.append(top_p_name)

        top_k = kwargs.pop(TOP_K_KWARG, None)
        if top_k is not None:
            top_k_name = "_".join((TOP_K_NAME, str(top_k)))
            if top_k_name not in self._post_processors:
                self._post_processors[top_k_name] = TopKLogitProcessor(top_k=top_k)
            post_processing.append(top_k_name)

        temp = kwargs.pop(TEMPERATURE_KWARG, None)
        if temp is not None:
            temp_name = "_".join((TEMP_NAME, str(temp)))
            if temp_name not in self._post_processors:
                self._post_processors[temp_name] = TemperatureLogitProcessor(
                    temperature=temp)
            post_processing.append(temp_name)

        do_sample = kwargs.pop(DO_SAMPLE_KWARG, DO_SAMPLE_DEFAULT)
        if do_sample:
            sampler_name = "_".join((SAMPLER_NAME, "logits"))
            if sampler_name not in self._post_processors:
                self._post_processors[sampler_name] = LogitsSampler()
        else:
            sampler_name = "_".join((SAMPLER_NAME, "greedy"))
            if sampler_name not in self._post_processors:
                self._post_processors[sampler_name] = GreedySampler()
        post_processing.append(sampler_name)

        stop = kwargs.pop(STOP_KWARG, None)
        if stop is not None:
            stop_name = "_".join((STOP_NAME, stop))
            if stop_name not in self._post_processors:
                self._post_processors[stop_name] = TokenStopCriterion(
                    token=stop,
                    tokenizer=self.tokenizer)
        else:
            stop_name = STOP_NAME
            if STOP_NAME not in self._post_processors:
                self._post_processors[stop_name] = EosGenerationStopCriterion(
                    tokenizer=self.tokenizer)
        post_processing.append(stop_name)

        assert kwargs == {}, f"Unknown keyword arguments {kwargs}"

        return RaggedRequest(
            tid=tid,
            uid=uid,
            input_tokens=input_tokens,
            prompt_tokens=input_tokens,
            seq_length=0,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            last_in_prompt=True,
            post_processing=post_processing,
            stream=stream,
            ignore_eos=ignore_eos,
            return_full_text=return_full_text,
        )

    def make_response(self,
                      generated_text: str,
                      prompt_length: int,
                      generated_length: int,
                      finish_reason: GenerationFinishReason) -> Response:
        return Response(generated_text=generated_text,
                        prompt_length=prompt_length,
                        generated_length=generated_length,
                        finish_reason=finish_reason)

    def put(self, uids: List[int], tokenized_input: List[torch.Tensor]) -> torch.Tensor:
        return self.inference_engine.put(uids, tokenized_input)

    def flush(self, uids: List[int]) -> None:
        for uid in uids:
            self.inference_engine.flush(uid)


class MIIPipeline(RaggedBatchBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tid = threading.get_ident()

    def __call__(self, inputs: Union[str, List[str]], **kwargs) -> ResponseBatch:
        if isinstance(inputs, str):
            inputs = [inputs]
        outputs: ResponseBatch = ResponseBatch([])
        uids_running: List[int] = list(range(len(inputs)))
        uids_complete_order: List[int] = []

        for uid, input in zip(uids_running, inputs):
            request_kwargs = kwargs.copy()
            self._put_request(uid, input, request_kwargs)

        self.schedule_requests()

        if self.is_rank_0:
            # Rank 0 runs generate() until all responses are returned
            while uids_running:
                self.generate()
                while not self.result_queues[self.tid].empty():
                    uid, response = self._get_response()
                    outputs.append(response)
                    self._queue_flush_request(uid)
                    uids_complete_order.append(uids_running.index(uid))
                    uids_running.remove(uid)
            # Ensure final flush requests broadcast and
            # kick ranks 1 -> n out of the while loop
            self._bcast_requests(force=True)
        else:
            # Ranks 1 -> n just run generate() until there are no more requests
            while self.scheduled_requests:
                self.generate()

        outputs = ResponseBatch([
            r for idx,
            r in sorted(zip(uids_complete_order,
                            outputs),
                        key=lambda pair: pair[0])
        ])

        if self.model_config.all_rank_output:
            outputs = self._bcast_responses(outputs)

        return outputs

    def _put_request(self, uid: int, input: str, kwargs: Dict[str, Any]) -> None:
        self.result_queues[self.tid] = queue.Queue()
        input_tokens = self.tokenizer.encode(input)
        request = self.make_request(self.tid, uid, input_tokens, kwargs)
        self.request_queue.put(request)

    def _get_response(self) -> Tuple[int, Response]:
        result = self.result_queues[self.tid].get()
        uid = result[0]
        generated_tokens = self.tokenizer.decode(result[1])
        response = self.make_response(generated_tokens, result[2], result[3], result[4])
        return uid, response

    def _bcast_responses(self, responses: ResponseBatch) -> ResponseBatch:
        if self.is_rank_0:
            data_dicts = [r.get_msg() for r in responses]
            json_data = ujson.dumps(data_dicts)
            self.socket.send_string(json_data)
        else:
            json_data = self.socket.recv_string()
            data_dicts = ujson.loads(json_data)
            responses = ResponseBatch([Response.from_msg(msg) for msg in data_dicts])
        return responses


class MIIAsyncPipeline(RaggedBatchBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uids = set()
        self.lock = threading.Lock()
        self.thread = None
        self.stop_thread = False
        self._is_shutdown = False
        self.UID_RANGE_LB = 1
        self.UID_RANGE_UB = 10000

    def __call__(self) -> None:
        # CUDA device gets reset, must set it again to avoid problems
        get_accelerator().set_device(int(os.getenv("LOCAL_RANK", "0")))
        while True:
            self.generate()

            if (self.stop_thread and self.request_queue.empty()
                    and all(q.empty() for q in self.result_queues.values())):
                break

    def _get_uid(self) -> int:
        with self.lock:
            uid = random.randrange(self.UID_RANGE_LB, self.UID_RANGE_UB)
            while uid in self.uids:
                uid = random.randrange(self.UID_RANGE_LB, self.UID_RANGE_UB)
            self.uids.add(uid)

        return uid

    def put_request(self, prompt: str, kwargs: Dict) -> int:
        # TODO: We should avoid any request/response work with non-rank 0, but
        # this requires some refactoring how we do the put and request in
        # `ModelResponse`
        #if not self.is_rank_0:
        #    return
        if self.stop_thread:
            raise RuntimeError("The request queue was shutdown.")

        uid = self._get_uid()

        # Temporary hack to avoid non-rank 0 processes not shutting down. See
        # related TODO above.
        if not self.is_rank_0:
            return uid

        tid = threading.get_ident()
        with self.lock:
            if tid not in self.result_queues:
                self.result_queues[tid] = queue.Queue()

        input_tokens = self.tokenizer.encode(prompt)
        request = self.make_request(tid, uid, input_tokens, kwargs)
        self.request_queue.put(request)

        return uid

    def is_response_ready(self, uid: int) -> bool:
        if not self.is_rank_0:
            return True
        return not self.result_queues[uid].empty()

    def get_response(self) -> Tuple[int, Response]:
        # TODO: We should avoid any request/response work with non-rank 0, but
        # this requires some refactoring how we do the put and request in
        # `ModelResponse`
        if not self.is_rank_0:
            return -1, Response(generated_text="",
                            prompt_length=None,
                            generated_length=None,
                            finish_reason=None)
        tid = threading.get_ident()
        result = self.result_queues[tid].get()
        uid = result[0]
        generated_token_ids = result[1]
        if len(generated_token_ids) == 0:
            generated_text = ""
        else:
            generated_text = self.tokenizer.decode(generated_token_ids)
        response = self.make_response(generated_text, result[2], result[3], result[4])
        return uid, response

    def start(self) -> None:
        self.thread = threading.Thread(target=self, daemon=True)
        self.thread.start()

    def shutdown(self) -> None:
        self.stop_thread = True
        self.thread.join()
        self._is_shutdown = True

    def is_shutdown(self) -> bool:
        return self._is_shutdown

    def flush_uid(self, uid: int) -> None:
        with self.lock:
            if self.is_rank_0:
                self._queue_flush_request(uid)
            self.uids.remove(uid)
