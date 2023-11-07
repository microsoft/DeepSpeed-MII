# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    from mii.batching.ragged_batching import RaggedRequestBatch


def run_batch_processing(input_tensor: torch.Tensor,
                         requests: "RaggedRequestBatch",
                         processor_fns: Dict[str,
                                             Any]) -> torch.Tensor:
    idx_list = []
    output_list = []
    for key, process_fn in processor_fns.items():
        idx = [i for i, r in enumerate(requests) if key in r.post_processing]
        if not idx:
            continue
        filtered_input = input_tensor[idx]
        idx_list.extend(idx)
        output_list.append(process_fn(filtered_input))
    if not output_list:
        return input_tensor
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
    return done_tokens
