# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import itertools
from collections import defaultdict
from typing import Any, Dict

import torch

from .generation.logit_processors import (
    TopKLogitProcessor,
    TopPLogitProcessor,
    TemperatureLogitProcessor,
    NucleusSamplingLogitProcessor,
)
from .generation.samplers import LogitsSampler, GreedySampler
from .generation.stop_criterion import (
    EosGenerationStopCriterion,
    NewLineDelimitedStopCriterion,
)

LOGITS_PROCESSORS = {
    "TopK": TopKLogitProcessor,
    "TopP": TopPLogitProcessor,
    "Temperature": TemperatureLogitProcessor,
    "NucleusSampling": NucleusSamplingLogitProcessor,
}

SAMPLERS = {"Logits": LogitsSampler, "Greedy": GreedySampler}

STOP_CRITERIA = {
    "EosGeneration": EosGenerationStopCriterion,
    "NewLineDelimited": NewLineDelimitedStopCriterion,
}

DEFAULT_LOGITS_PROCESSOR = {"name": "TopP", "args": {"top_p": 0.9}}
DEFAULT_SAMPLER = {"name": "Logits"}
DEFAULT_STOP_CRITERION = {"name": "EosGeneration"}


def _create_postprocessor(config: Dict[str,
                                       Any],
                          classes: Dict[str,
                                        Any],
                          default_args: Dict[str,
                                             Any] = {}):
    assert "name" in config

    name = config["name"]
    if name not in classes:
        raise ValueError(f"Unknown postprocessor {name}")
    args = config["args"] if "args" in config else {}
    args.update(default_args)
    return classes[name](**args)


def _run_batch_postprocess(input_tensor,
                           requests,
                           get_processor_fn,
                           get_result_fn=lambda x: x):
    processor_map = {
        get_processor_fn(r).get_key(): get_processor_fn(r)
        for r in requests
    }
    processor_indices = defaultdict(list)

    for i, r in enumerate(requests):
        key = get_processor_fn(r).get_key()
        processor_indices[key].append(i)

    indice_list = []
    outputs_list = []
    for key, indices in processor_map.items():
        processor = processor_map[key]
        indices = processor_indices[key]
        input_filtered = input_tensor[indices]
        output_filtered = get_result_fn(processor(input_filtered))
        indice_list.append(indices)
        outputs_list.append(output_filtered)

    indice = list(itertools.chain.from_iterable(indice_list))
    outputs = torch.cat(outputs_list, dim=0)
    return outputs[torch.argsort(torch.tensor(indice))]


def run_batch_logit_processor(input_tensor, requests):
    return _run_batch_postprocess(input_tensor, requests, lambda r: r.logit_processor)


def run_batch_sampler(input_tensor, requests):
    return _run_batch_postprocess(input_tensor,
                                  requests,
                                  lambda r: r.sampler,
                                  lambda x: x[0])


def run_batch_stop_criterion(input_tensor, requests):
    return _run_batch_postprocess(input_tensor, requests, lambda r: r.stop_criterion)
