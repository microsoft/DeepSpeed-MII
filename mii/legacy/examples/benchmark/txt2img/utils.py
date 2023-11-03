# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import time
import deepspeed
import mii
import numpy
import diffusers
import transformers

from packaging import version

assert version.parse(diffusers.__version__) >= version.parse('0.7.1'), "diffusers must be 0.7.1+"
assert version.parse(mii.__version__) >= version.parse("0.0.3"), "mii must be 0.0.3+"
assert version.parse(deepspeed.__version__) >= version.parse("0.7.5"), "deepspeed must be 0.7.5+"
assert version.parse(transformers.__version__) >= version.parse("4.24.0"), "transformers must be 4.24.0+"


def benchmark(func, inputs, save_path=".", trials=5, tag="", save=True):
    # Turn off the tqdm progress bar
    if hasattr(func, "set_progress_bar_config"):
        func.set_progress_bar_config(disable=True)

    durations = []
    for trial in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            results = func(inputs)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        durations.append(duration)
        print(f"trial={trial}, time_taken={duration:.4f}")
        if save:
            for idx, img in enumerate(results.images):
                img.save(os.path.join(save_path, f"{tag}-trial{trial}-img{idx}.png"))
    print(f"median duration: {numpy.median(durations):.4f}")
