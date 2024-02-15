# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import copy
import gc
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from collections import deque, defaultdict
from functools import cached_property
from typing import Dict, Tuple, List, Any, Union, DefaultDict

import torch
import ujson
import zmq
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.timer import SynchronizedWallClockTimer

from mii.batching.constants import TOP_K_NAME, TOP_P_NAME, TEMP_NAME, SAMPLER_NAME, STOP_NAME
from mii.batching.data_classes import Response, Request, RequestBatch
from mii.batching.generation.logit_processors import TopPLogitProcessor, TopKLogitProcessor, TemperatureLogitProcessor
from mii.batching.generation.samplers import LogitsSampler, GreedySampler
from mii.batching.generation.stop_criterion import EosGenerationStopCriterion, TokenStopCriterion
from mii.batching.postprocess import (
    run_batch_logit_processing,
    run_batch_sampler,
    run_batch_stop_criterion,
)
from mii.batching.utils import sync_debug, profiler
from mii.config import GenerateParamsConfig
from mii.constants import GenerationFinishReason, ZMQ_RECV_TIMEOUT
from mii.logging import logger
from mii.modeling.tokenizers import MIITokenizerWrapper


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
        self.scheduled_requests: RequestBatch = RequestBatch()
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

        self._zmq_context = zmq.Context()
        torch.cuda.synchronize()
        if self.is_rank_0:
            self.socket = self._zmq_context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.zmq_port}")
            time.sleep(1)  # Give the subscriber a change to connect
        else:
            self.socket = self._zmq_context.socket(zmq.SUB)
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
    def _bcast_requests(self, force=False) -> RequestBatch:
        if self.is_rank_0:
            if not self.scheduled_requests and not force:
                return self.scheduled_requests
            # Rank 0 gets batch of requests and broadcasts to other ranks
            data_dicts = self.scheduled_requests.to_msg_dicts()
            json_data = ujson.dumps(data_dicts)
            self.socket.send_string(json_data)
        else:
            try:
                json_data = self.socket.recv_string()
                data_dicts = ujson.loads(json_data)
                self.scheduled_requests = RequestBatch.from_msg_dicts(data_dicts)
            except zmq.Again:
                self.scheduled_requests = RequestBatch()

        return self.scheduled_requests

    def _reset_scheduler_bookkeeping(self) -> None:
        self.scheduled_requests = RequestBatch()
        self.scheduled_length = 0
        self.scheduled_seq_num = 0
        self.scheduled_req_blocks = 0

    @sync_debug
    def _process_logits(
            self,
            next_token_logits: torch.Tensor,
            running_requests: RequestBatch) -> Tuple[torch.Tensor,
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
        done_tokens = done_tokens.to(torch.device("cpu"), non_blocking=False)
        return next_tokens, done_tokens

    @sync_debug
    def _generate_output(self, r: Request) -> bool:
        outputs = []
        if r.stream:
            outputs.append((
                r.uid,
                [r.next_token],
                r.prompt_length,
                r.num_generated_tokens,
                GenerationFinishReason.NONE,
                r.stream,
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
                r.stream,
            ))
        for output in outputs:
            self.result_queues[r.tid].put_nowait(output)

    def _schedule_token_gen(self, requests: List[Request]) -> None:
        free_blocks = min(self.inference_engine.free_blocks)
        conf_manager = self.inference_engine._config.state_manager

        num_schedulable = min([
            len(requests),
            conf_manager.max_ragged_sequence_count,
            conf_manager.max_ragged_batch_size
        ])

        for r in requests[:num_schedulable]:
            block_capacity = self.inference_engine.get_remaining_block_capacity(r.uid)
            # We can schedule token generation if the last block has a capacity
            if block_capacity > 0:
                self.scheduled_length += 1
                self.scheduled_requests.append(r)
            elif free_blocks > 0:
                # We need a new block
                free_blocks -= 1
                self.scheduled_length += 1
                self.scheduled_req_blocks += 1
                self.scheduled_requests.append(r)

    def _schedule_prompts(self, requests: List[Request]) -> None:
        free_blocks = min(self.inference_engine.free_blocks)
        conf_manager = self.inference_engine._config.state_manager

        for r in requests:
            if free_blocks == 0:
                break

            if r.max_length <= r.seq_length:
                continue

            # Make sure that the engine has enough capacity to process the batch
            if len(self.scheduled_requests) > conf_manager.max_ragged_sequence_count:
                break

            max_batch_size = conf_manager.max_ragged_batch_size - self.scheduled_length
            if max_batch_size <= 0:
                break

            max_blocks = free_blocks - self.scheduled_req_blocks

            if len(r.input_tokens) > 1:
                # When the KV cache is out of capacity, we release KV cache blocks for a request.
                # However, we can immediately schedule the request again if we split the request.
                # So we make sure that we have capacity for the entire prompt (+tokens already generated).
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

        next_token_gen_reqs = []
        prompt_reqs = []

        for r in self.buffer:
            if r.is_flush_request:
                self.scheduled_requests.append(r)
            else:
                if r.num_generated_tokens > 0:
                    if r.max_length > r.seq_length:
                        next_token_gen_reqs.append(r)
                else:
                    prompt_reqs.append(r)

        # We want to process next token generation first
        self._schedule_token_gen(next_token_gen_reqs)
        self._schedule_prompts(prompt_reqs)

        if len(self.buffer) > 0 and len(self.scheduled_requests) == 0:
            self.scheduled_requests = RequestBatch()
            self.reset_request_status()
        else:
            scheduled_requests_ids = set(id(r) for r in self.scheduled_requests)
            self.buffer = deque(
                [r for r in self.buffer if id(r) not in scheduled_requests_ids])

    def _queue_flush_request(self, uid: int) -> None:
        self.request_queue.put_nowait(
            Request(
                tid=None,
                uid=uid,
                input_tokens=None,
                prompt_tokens=None,
                seq_length=None,
                last_in_prompt=None,
                post_processing=None,
                generate_params=None,
            ))

    def reset_request_status(self):
        ## Get the last request that consumes KV cache
        last_r = None
        for r in self.buffer:
            if r.seq_length > 0:
                last_r = r
        assert last_r is not None, "Function to clear the KV cache is invoked, but no request consumes KV cache"

        ## Schedule flushing r
        self.scheduled_requests.append(
            Request(
                tid=None,
                uid=last_r.uid,
                input_tokens=None,
                prompt_tokens=None,
                seq_length=None,
                last_in_prompt=None,
                post_processing=None,
                generate_params=None,
            ))

        ## Rebuild the request
        new_req = copy.copy(last_r)
        new_req.prompt_tokens = new_req.input_tokens = torch.concat(
            [last_r.prompt_tokens] + [t.unsqueeze(0) for t in last_r.generated_tokens])
        new_req.seq_length = 0
        new_req.max_new_tokens = last_r.max_new_tokens - len(last_r.generated_tokens)
        new_req.clear_generated_token()

        ## Remove the requests from buffer and queue
        new_buffer = deque()
        for r in self.buffer:
            if r.uid != last_r.uid:
                new_buffer.append(r)

        while not self.request_queue.empty():
            r = self.request_queue.get_nowait()
            if r.uid != last_r.uid:
                new_buffer.append(r)
        new_buffer.append(new_req)
        self.buffer = new_buffer

    def make_request(self,
                     tid: int,
                     uid: int,
                     input_tokens: torch.Tensor,
                     kwargs: Dict) -> Request:
        kwargs["prompt_length"] = len(input_tokens)
        kwargs["max_length"] = kwargs.get("max_length", self.max_length)
        generate_params = GenerateParamsConfig(**kwargs)

        post_processing = []

        top_p = generate_params.top_p
        top_p_name = "_".join((TOP_P_NAME, str(top_p)))
        if top_p_name not in self._post_processors:
            self._post_processors[top_p_name] = TopPLogitProcessor(top_p=top_p)
        post_processing.append(top_p_name)

        top_k = generate_params.top_k
        if top_k is not None:
            top_k_name = "_".join((TOP_K_NAME, str(top_k)))
            if top_k_name not in self._post_processors:
                self._post_processors[top_k_name] = TopKLogitProcessor(top_k=top_k)
            post_processing.append(top_k_name)

        temp = generate_params.temperature
        if temp is not None:
            temp_name = "_".join((TEMP_NAME, str(temp)))
            if temp_name not in self._post_processors:
                self._post_processors[temp_name] = TemperatureLogitProcessor(
                    temperature=temp)
            post_processing.append(temp_name)

        do_sample = generate_params.do_sample
        if do_sample:
            sampler_name = "_".join((SAMPLER_NAME, "logits"))
            if sampler_name not in self._post_processors:
                self._post_processors[sampler_name] = LogitsSampler()
        else:
            sampler_name = "_".join((SAMPLER_NAME, "greedy"))
            if sampler_name not in self._post_processors:
                self._post_processors[sampler_name] = GreedySampler()
        post_processing.append(sampler_name)

        stop = generate_params.stop
        if stop != []:
            for each_stop in stop:
                stop_name = STOP_NAME + '_' + each_stop
                if stop_name not in self._post_processors:
                    self._post_processors[stop_name] = TokenStopCriterion(
                        token=each_stop,
                        tokenizer=self.tokenizer)
                post_processing.append(stop_name)
        else:
            stop_name = STOP_NAME
            if STOP_NAME not in self._post_processors:
                self._post_processors[stop_name] = EosGenerationStopCriterion(
                    tokenizer=self.tokenizer)
            post_processing.append(stop_name)

        return Request(
            tid=tid,
            uid=uid,
            input_tokens=input_tokens,
            prompt_tokens=input_tokens,
            seq_length=0,
            last_in_prompt=True,
            post_processing=post_processing,
            generate_params=generate_params,
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
        # Call inference engine. You can skip checking schedulability because we already checked when scheduling
        return self.inference_engine.put(uids, tokenized_input, do_checks=False)

    def flush(self, uids: List[int]) -> None:
        for uid in uids:
            self.inference_engine.flush(uid)


@dataclass
class StreamState:
    prev_token_size: int
    token_ids: List[int]


class ReadableStream:
    def __init__(self, tokenizer: MIITokenizerWrapper) -> None:
        self.tokenizer = tokenizer
        self.stream_state: Dict[int, StreamState] = {}

    def init_state(self, thread_id: int) -> StreamState:
        if thread_id not in self.stream_state:
            self.stream_state[thread_id] = StreamState(token_ids=[], prev_token_size=0)
            return self.stream_state[thread_id]
        return self.stream_state[thread_id]

    def flush_state(self, thread_id: int) -> None:
        if thread_id in self.stream_state:
            del self.stream_state[thread_id]

    def decode(self, thread_id: int, token_ids: List[int]) -> str:
        state = self.init_state(thread_id)
        output = []

        for token_id in token_ids:
            state.token_ids.append(token_id)
            decoded = self.tokenizer.decode(state.token_ids)

            # We don't have enough token_ids in the buffer and
            # tokenizer returned unicode 'U+FFFD REPLACEMENT CHARACTER'
            if "\ufffd" in decoded:
                continue

            if state.prev_token_size > 0:
                prev_token = state.token_ids[:state.prev_token_size]
                state.token_ids = state.token_ids[state.prev_token_size:]
                decoded = decoded.replace(self.tokenizer.decode(prev_token), "", 1)

            output.append(decoded)
            state.prev_token_size = len(state.token_ids)

        return "".join(output)


class MIIPipeline(RaggedBatchBase):
    """
    Pipeline class that inherits from :class:`RaggedBatchBase` and provides
    functionality of ragged batching and dynamic splitfuse. This class is
    returned from :func:`mii.pipeline`.
    """
    def __init__(self, all_rank_output: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tid = threading.get_ident()
        self._all_rank_output = all_rank_output
        self._destroyed = False

    def __call__(self,
                 prompts: Union[str,
                                List[str]],
                 **generate_kwargs) -> List[Response]:
        """
        Generates text for the given prompts

        :param prompts: The string or list of strings used as prompts for generation.
        :param \**generate_kwargs: Generation keywords. A full list can be found
            in :class:`GenerateParamsConfig <mii.config.GenerateParamsConfig>`.

        :return: A list of :class:`Response` objects containing the generated
            text for all prompts.
        """ # noqa: W605
        if self._destroyed:
            raise RuntimeError(
                "The inference engine of this pipeline has been destroyed.")

        if isinstance(prompts, str):
            prompts = [prompts]
        outputs: List[Response] = []
        uids_running: List[int] = list(range(len(prompts)))
        uids_complete_order: List[int] = []

        for uid, input in zip(uids_running, prompts):
            request_kwargs = generate_kwargs.copy()
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
                    uids_complete_order.append(uid)
                    uids_running.remove(uid)
            # Ensure final flush requests broadcast and
            # kick ranks 1 -> n out of the while loop
            self._bcast_requests(force=True)
        else:
            # Ranks 1 -> n just run generate() until there are no more requests
            while self.scheduled_requests:
                self.generate()

        outputs = [
            r for idx,
            r in sorted(zip(uids_complete_order,
                            outputs),
                        key=lambda pair: pair[0])
        ]

        if self._all_rank_output:
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

    def _bcast_responses(self, responses: List[Response]) -> List[Response]:
        if self.is_rank_0:
            data_dicts = [r.to_msg_dict() for r in responses]
            json_data = ujson.dumps(data_dicts)
            self.socket.send_string(json_data)
        else:
            json_data = self.socket.recv_string()
            data_dicts = ujson.loads(json_data)
            responses = [Response.from_msg_dict(msg) for msg in data_dicts]
        return responses

    def destroy(self) -> None:
        del self.inference_engine
        self.socket.close()
        self._zmq_context.term()
        gc.collect()
        get_accelerator().empty_cache()
        self._destroyed = True


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
        self.readable_stream = ReadableStream(self.tokenizer)

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
        uid, generated_token_ids, prompt_length, generated_length, finish_reason, streaming = self.result_queues[tid].get()

        if len(generated_token_ids) == 0:
            generated_text = ""
            self.readable_stream.flush_state(tid)
        elif streaming:
            generated_text = self.readable_stream.decode(tid, generated_token_ids)
        else:
            generated_text = self.tokenizer.decode(generated_token_ids)

        response = self.make_response(
            generated_text=generated_text,
            prompt_length=prompt_length,
            generated_length=generated_length,
            finish_reason=finish_reason,
        )
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
