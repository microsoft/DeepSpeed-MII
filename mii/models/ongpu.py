from typing import Callable
import time
import psutil
import os
import torch
import argparse
from torch import Tensor


def see_memory_usage(tag):
    print(f"-----{tag}----")
    print(
        f"GPU Memory allocated: {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
            Max memory allocated: {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB"
    )

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    print(f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')
    print(f"-----{tag}----")


class OnGPU(object):
    _orig_torch_empty = torch.empty
    _orig_torch_zeros = torch.zeros
    _orig_torch_ones = torch.ones
    _orig_torch_full = torch.full

    def __init__(self, dtype, enabled=True):
        self.dtype = dtype
        self.enabled = enabled

    @staticmethod
    def fp_tensor_constructor(fn: Callable, target_fp_dtype: torch.dtype) -> Callable:
        def wrapped_fn(*args, **kwargs) -> Tensor:
            if kwargs.get("device", None) is None:
                kwargs[
                    'device'] = 'meta'  #torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
            tensor: Tensor = fn(*args, **kwargs)
            if tensor.is_floating_point():
                tensor = tensor.to(target_fp_dtype)
            return tensor

        return wrapped_fn

    @staticmethod
    def get_new_tensor_fn_for_dtype(dtype: torch.dtype) -> Callable:
        def new_tensor(cls, *args) -> Tensor:
            device = 'meta'  #torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
            tensor = self._orig_torch_empty(0, device=device).new_empty(*args)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype)
            return tensor

        return new_tensor

    def __enter__(self):
        if not self.enabled:
            return
        torch.Tensor.__old_new__ = torch.Tensor.__new__
        torch.Tensor.__new__ = self.get_new_tensor_fn_for_dtype(self.dtype)
        torch.empty = self.fp_tensor_constructor(self._orig_torch_empty, self.dtype)
        torch.zeros = self.fp_tensor_constructor(self._orig_torch_zeros, self.dtype)
        torch.ones = self.fp_tensor_constructor(self._orig_torch_ones, self.dtype)
        torch.full = self.fp_tensor_constructor(self._orig_torch_full, self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return
        torch.Tensor.__new__ = torch.Tensor.__old_new__
        torch.empty = self._orig_torch_empty
        torch.zeros = self._orig_torch_zeros
        torch.ones = self._orig_torch_ones
        torch.full = self._orig_torch_full
