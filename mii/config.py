# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from typing import Union, List
from enum import Enum
from pydantic import BaseModel, validator, root_validator
from deepspeed.launcher.runner import DLTS_HOSTFILE


class DtypeEnum(Enum):
    # The torch dtype must always be the first value (so we return torch.dtype)
    fp16 = torch.float16, "torch.float16", "fp16", "float16", "half"
    bf16 = torch.bfloat16, "torch.bfloat16", "bf16", "bfloat16"
    fp32 = torch.float32, "torch.float32", "fp32", "float32", "float"
    int8 = torch.int8, "torch.int8", "int8"

    # Copied from https://stackoverflow.com/a/43210118
    # Allows us to use multiple values for each Enum index and returns first
    # listed value when Enum is called
    def __new__(cls, *values):
        obj = object.__new__(cls)
        # first value is canonical value
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj

    def __repr__(self):
        return "<%s.%s: %s>" % (
            self.__class__.__name__,
            self._name_,
            ", ".join([repr(v) for v in self._all_values]),
        )


class MIIConfig(BaseModel):
    tensor_parallel: int = 1
    port_number: int = 50050
    dtype: DtypeEnum = torch.float32
    meta_tensor: bool = False
    load_with_sys_mem: bool = False
    enable_cuda_graph: bool = False
    checkpoint_dict: Union[dict, None] = None
    deploy_rank: Union[int, List[int]] = -1
    torch_dist_port: int = 29500
    hf_auth_token: str = None
    replace_with_kernel_inject: bool = True
    profile_model_time: bool = False
    skip_model_check: bool = False
    max_tokens: int = 1024
    enable_restful_api: bool = False
    restful_api_port: int = 51080
    replica_num: int = 1
    hostfile: str = DLTS_HOSTFILE
    trust_remote_code: bool = False

    @validator("deploy_rank")
    def deploy_valid(cls, field_value, values):
        if "tensor_parallel" not in values:
            raise ValueError(
                "'tensor_parallel' must be defined in the pydantic model before 'deploy_rank'"
            )

        # if deploy rank is not given, default to align with TP value
        if field_value == -1:
            field_value = list(range(values["tensor_parallel"]))

        # ensure deploy rank type is always list for easier consumption later
        if not isinstance(field_value, list):
            field_value = [field_value]

        # number of ranks provided must be equal to TP size, DP is handled outside MII currently
        assert values["tensor_parallel"] == len(field_value), \
            f"{len(field_value)} rank(s) provided in 'deploy_rank' does not align with tensor_parallel size of {values['tensor_parallel']}"
        return field_value

    @validator('checkpoint_dict')
    def checkpoint_dict_valid(cls, value):
        if value is None:
            return value
        if value.get('base_dir', ''):
            raise ValueError(
                "please unset 'base_dir' it will be set w.r.t. the deployment 'model_path'"
            )
        for k in ['checkpoints', 'parallelization', 'version', 'type']:
            if not value.get(k, ''):
                raise ValueError(f"Missing key={k} in checkpoint_dict")
        return value

    @root_validator
    def meta_tensor_or_sys_mem(cls, values):
        if values.get("meta_tensor") and values.get("load_with_sys_mem"):
            raise ValueError(
                "`meta_tensor` and `load_with_sys_mem` cannot be active at the same time."
            )
        return values

    class Config:
        validate_all = True
        validate_assignment = True
        use_enum_values = True
        extra = 'forbid'
        json_encoders = {torch.dtype: lambda x: str(x)}


class ReplicaConfig(BaseModel):
    task: str = ""
    deployment_name: str = ""
    hostname: str = ""
    tensor_parallel_ports: List[int] = []
    torch_dist_port: int = None
    gpu_indices: List[int] = []

    class Config:
        validate_all = True
        validate_assignment = True


class LoadBalancerConfig(BaseModel):
    port: int = None
    replica_configs: List[ReplicaConfig] = []

    class Config:
        validate_all = True


validate_assignment = True


class DeploymentConfig(BaseModel):
    deployment_name: str
    task: str
    model: str
    enable_deepspeed: bool = True
    enable_zero: bool = False
    GPU_index_map: dict = None
    mii_config: MIIConfig = MIIConfig.parse_obj({})
    ds_config: dict = None
    version: int = 1
