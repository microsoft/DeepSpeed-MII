import os
import torch
import time
import deepspeed
#import mii
import diffusers

from packaging import version

assert version.parse(diffusers.__version__) >= version.parse('0.6.0'), "diffusers must be 0.6.0+"
# TODO: add __version__ support into mii
#assert version.parse(mii.__version__) >= version.parse("0.0.3"), "mii must be 0.0.3+"
assert version.parse(deepspeed.__version__) >= version.parse("0.7.4"), "deepspeed must be 0.7.4+"


def benchmark(func, inputs, save_path=".", trials=5, tag=""):
    for trial in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        results = func(inputs)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        print(f"trial={trial}, time_taken={duration:.4f}")
        for idx, img in enumerate(results.images):
            img.save(os.path.join(save_path, f"{tag}-trial{trial}-img{idx}.png"))
