# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from typing import Any, Union, List, Optional
from enum import Enum
from pydantic import field_validator, model_validator, ConfigDict, BaseModel, FieldValidationInfo

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
    checkpoint_dict: Optional[dict] = None
    deploy_rank: Union[int, List[int]] = -1
    torch_dist_port: int = 29500
    hf_auth_token: Optional[str] = None
    replace_with_kernel_inject: bool = True
    profile_model_time: bool = False
    skip_model_check: bool = False
    max_tokens: int = 1024
    enable_restful_api: bool = False
    restful_api_port: int = 51080
    replica_num: int = 1
    hostfile: str = DLTS_HOSTFILE
    trust_remote_code: bool = False

    @field_validator('checkpoint_dict')
    @classmethod
    def checkpoint_dict_valid(cls, v: Union[None, dict], info: FieldValidationInfo):
        if v is None:
            return v
        if v.get('base_dir', ''):
            raise ValueError(
                "please unset 'base_dir' it will be set w.r.t. the deployment 'model_path'"
            )
        for k in ['checkpoints', 'parallelization', 'version', 'type']:
            if not v.get(k, ''):
                raise ValueError(f"Missing key={k} in checkpoint_dict")
        return v

    @model_validator(mode="after")
    @classmethod
    def deploy_rank_valid(cls, data: Any) -> Any:
        # if deploy rank is not given, default to align with TP value
        if data.deploy_rank == -1:
            data.deploy_rank = list(range(data.tensor_parallel))

        # ensure deploy rank type is always list for easier consumption later
        if not isinstance(data.deploy_rank, list):
            data.deploy_rank = [data.deploy_rank]

        # number of ranks provided must be equal to TP size, DP is handled outside MII currently
        assert data.tensor_parallel == len(data.deploy_rank), \
            f"{len(data.deploy_rank)} rank(s) provided in 'deploy_rank' does not align with tensor_parallel size of {data.tensor_parallel}"
        return data

    @model_validator(mode="after")
    @classmethod
    def meta_tensor_or_sys_mem(cls, data: Any) -> Any:
        if data.meta_tensor and data.load_with_sys_mem:
            raise ValueError(
                "`meta_tensor` and `load_with_sys_mem` cannot be active at the same time."
            )
        return data

    # TODO[pydantic]: The following keys were removed: `json_encoders`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
    model_config = ConfigDict(validate_default=True,
                              validate_assignment=True,
                              use_enum_values=True,
                              extra='forbid',
                              json_encoders={torch.dtype: lambda x: str(x)})


class ReplicaConfig(BaseModel):
    hostname: str = ""
    tensor_parallel_ports: List[int] = []
    torch_dist_port: int = None
    gpu_indices: List[int] = []
    model_config = ConfigDict(validate_default=True, validate_assignment=True)


class LoadBalancerConfig(BaseModel):
    port: int = None
    replica_configs: List[ReplicaConfig] = []
    model_config = ConfigDict(validate_default=True, validate_assignment=True)
