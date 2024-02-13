# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from typing import TYPE_CHECKING, Any, Dict, List

import torch

if TYPE_CHECKING:
    from mii.batching.ragged_batching import RaggedRequestBatch


def run_batch_processing(input_tensor: torch.Tensor,
                         requests: "RaggedRequestBatch",
                         processor_fns: Dict[str,
                                             Any]) -> torch.Tensor:
    idx_list: List[int] = []
    output_list: List[torch.Tensor] = []

    # Apply all the post-processing functions
    for key, process_fn in processor_fns.items():

        # Get the index of tensors that need to be processed
        idx = [i for i, r in enumerate(requests) if key in r.post_processing]
        if not idx:
            # Short circuit if there is not work to do
            continue

        # Run post processing on the filtered inputs
        filtered_input = input_tensor[idx]
        idx_list.extend(idx)
        output_list.append(process_fn(filtered_input))

    # If there was no work done, return the input tensor
    if not output_list:
        return input_tensor

    # If there are unprocessed requests, append them to the output
    unprocessed_idx = list(set(range(len(requests))).difference(idx_list))
    if unprocessed_idx:
        idx_list.append(unprocessed_idx)
        output_list.append(input_tensor[unprocessed_idx])

    # Concatenate and return the output
    output = torch.cat(output_list, dim=0)
    return output[torch.argsort(torch.tensor(idx_list))]


def run_batch_logit_processing(input_logits: torch.Tensor,
                               requests: "RaggedRequestBatch",
                               processor_map: Dict[str,
                                                   Any]) -> torch.Tensor:
    top_k_fns = {k: v for k, v in processor_map.items() if "TopK" in k}
    top_p_fns = {k: v for k, v in processor_map.items() if "TopP" in k}
    temp_fns = {k: v for k, v in processor_map.items() if "Temp" in k}

    # Apply TopK, TopP, and Temperature in sequence
    output_logits = input_logits
    for fns in (top_k_fns, top_p_fns, temp_fns):
        output_logits = run_batch_processing(output_logits, requests, fns)
    return output_logits


def run_batch_sampler(input_logits: torch.Tensor,
                      requests: "RaggedRequestBatch",
                      processor_map: Dict[str,
                                          Any]) -> torch.Tensor:
    sampler_fns = {k: v for k, v in processor_map.items() if "Sampler" in k}
    next_tokens = run_batch_processing(input_logits, requests, sampler_fns)
    return next_tokens


def run_batch_stop_criterion(next_tokens: torch.Tensor,
                             requests: "RaggedRequestBatch",
                             processor_map: Dict[str,
                                                 Any]) -> torch.Tensor:
    stop_fns = {k: v for k, v in processor_map.items() if "Stop" in k}
    done_tokens = run_batch_processing(next_tokens, requests, stop_fns)
    done_tokens = torch.any(done_tokens.view((len(requests), -1)), dim=1)

    return done_tokens
