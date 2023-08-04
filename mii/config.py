# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from typing import Union, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, validator, root_validator, Field
#from deepspeed.launcher.runner import DLTS_HOSTFILE
DLTS_HOSTFILE = "dummy"
import mii
from mii.constants import (
    DEPLOYMENT_NAME_KEY,
    TASK_NAME_KEY,
    MODEL_NAME_KEY,
    ENABLE_DEEPSPEED_KEY,
    ENABLE_DEEPSPEED_ZERO_KEY,
    GPU_INDEX_KEY,
    DEEPSPEED_CONFIG_KEY,
    VERSION_KEY,
    MII_MODEL_PATH_DEFAULT,
)


class DeploymentType(Enum):
    LOCAL = "local"
    AML = "aml"
    NON_PERSISTENT = "non-persistent"


class Tasks(Enum):
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    QUESTION_ANSWERING = "question-answering"
    FILL_MASK = "fill-mask"
    TOKEN_CLASSIFICATION = "token-classification"
    CONVERSATIONAL = "conversational"
    TEXT2IMG = "text-to-image"


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


class MIIConfigModel(BaseModel):
    class Config:
        allow_population_by_field_name = True
        validate_all = True
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"
        json_encoders = {torch.dtype: lambda x: str(x)}


class ReplicaConfig(MIIConfigModel):
    task: Tasks = ""
    deployment_name: str = ""
    hostname: str = ""
    tensor_parallel_ports: List[int] = []
    torch_dist_port: Optional[int] = None
    gpu_indices: List[int] = []


class DeploymentConfig(MIIConfigModel):
    deployment_name: str = Field(alias=DEPLOYMENT_NAME_KEY)
    task: Tasks = Field(alias=TASK_NAME_KEY)
    model: str = Field(alias=MODEL_NAME_KEY)
    model_path: str = MII_MODEL_PATH_DEFAULT  # TODO or "model" if AML deployment
    ds_optimize: bool = Field(default=True, alias=ENABLE_DEEPSPEED_KEY)
    ds_zero: bool = Field(default=False, alias=ENABLE_DEEPSPEED_ZERO_KEY)
    GPU_index_map: List[dict] = Field([], alias=GPU_INDEX_KEY)
    ds_config: dict = Field(default={}, alias=DEEPSPEED_CONFIG_KEY)
    tensor_parallel: int = Field(1, gt=0)
    dtype: DtypeEnum = torch.float32
    meta_tensor: bool = False
    load_with_sys_mem: bool = False
    replace_with_kernel_inject: bool = True
    profile_model_time: bool = False
    skip_model_check: bool = False
    max_tokens: int = 1024
    enable_restful_api: bool = False
    replica_num: int = 1
    replica_configs: List[ReplicaConfig] = []
    deploy_rank: Union[int, List[int]] = -1
    enable_cuda_graph: bool = False
    checkpoint_dict: Union[dict, None] = None
    hf_auth_token: str = None
    trust_remote_code: bool = False

    @validator("checkpoint_dict")
    def checkpoint_dict_valid(cls, value):
        if value is None:
            return value
        if value.get("base_dir", ""):
            raise ValueError(
                "please unset 'base_dir' it will be set w.r.t. the deployment 'model_path'"
            )
        for k in ["checkpoints", "parallelization", "version", "type"]:
            if not value.get(k, ""):
                raise ValueError(f"Missing key={k} in checkpoint_dict")
        return value

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
        assert values["tensor_parallel"] == len(
            field_value
        ), f"{len(field_value)} rank(s) provided in 'deploy_rank' does not align with tensor_parallel size of {values['tensor_parallel']}"
        return field_value

    @root_validator
    def meta_tensor_or_sys_mem(cls, values):
        if values.get("meta_tensor") and values.get("load_with_sys_mem"):
            raise ValueError(
                "`meta_tensor` and `load_with_sys_mem` cannot be active at the same time."
            )
        return values

    @root_validator
    def zero_dtype_valid(cls, values):
        if values.get("ds_zero"):
            if values.get("ds_config").get("fp16", {}).get("enabled", False):
                assert (
                    values.get("dtype") == torch.float16
                ), "ZeRO FP16 enabled, `dtype` must be set to `torch.float16`"
            else:
                assert (
                    values.get("dtype") == torch.float16
                ), "ZeRO FP16 disabled, `dtype` must be set to `torch.float32`"
        return values

    @root_validator
    def deepspeed_or_zero(cls, values):
        assert not (
            values.get("ds_optimize") and values.get("ds_zero")
        ), "DeepSpeed and ZeRO cannot both be enabled, select only one"
        return values

    @root_validator
    def validator_model_and_task(cls, fields):
        if not fields.get("skip_model_check"):
            mii.utils.check_if_task_and_model_is_valid(fields.get("task"),
                                                       fields.get("model"))
            if fields.get("ds_optimize"):
                mii.utils.check_if_task_and_model_is_supported(
                    fields.get("task"),
                    fields.get("model"))
        return fields


class MIIConfig(MIIConfigModel):
    deployment_tag: str
    deployment_configs: List[DeploymentConfig]
    deployment_type: Any = DeploymentType.LOCAL
    enable_restful_api: bool = False
    restful_api_port: int = 51080
    port_number: int = 50050
    torch_dist_port: int = 29500
    hostfile: str = DLTS_HOSTFILE
    version: int = Field(default=1, alias=VERSION_KEY)

    @validator("deployment_configs", pre=True)
    def make_deployment_config_list(cls, field_value):
        if isinstance(field_value, dict):
            field_value = [field_value]
        return field_value

    @root_validator
    def validate_AML_deployment(cls, fields):
        if fields.get("deployment_type") == DeploymentType.AML:
            assert (
                len(fields.get("deployment_configs")) == 1
            ), "MII does not support empty/multi-model deployments on AML."
            allowed_chars = set(string.ascii_lowercase + string.ascii_uppercaes +
                                string.digits + "-")
            assert (
                set(fields.get("deployment_tag")) <= allowed_chars
            ), "AML deployment names can only contain a-z, A-Z, 0-9, and '-'."
        return fields

    @root_validator
    def set_deployment_tag(cls, fields):
        num_deployments = len(fields.get("deployment_configs"))
        if num_deployments == 0:
            assert fields.get(
                "deployment_tag"
            ), "Must set `deployment_tag` for empty deployment."
        elif num_deployments > 1:
            assert fields.get(
                "deployment_tag"
            ), "Must set `deployment_tag` for multi-model deployment."
        elif not fields.get("deployment_tag"):
            fields["deployment_tag"] = fields["deployment_configs"][0].deployment_name
        return fields

    @root_validator
    def create_replica_config(cls, fields):
        cls._used_tp_ports = [fields.get("port_number")]
        cls._used_dist_ports = [fields.get("torch_dist_port")]
        for dep_conf in fields.get("deployment_configs"):
            if not dep_conf.replica_configs:
                base_gpu_index = 0
                for i in range(dep_conf.replica_num):
                    # Get GPU indicies
                    if dep_conf.GPU_index_map:
                        gpu_indices = dep_conf.GPU_index_map[i]
                    else:
                        gpu_indices = list(
                            range(base_gpu_index,
                                  base_gpu_index + dep_conf.tensor_parallel))
                        base_gpu_index += dep_conf.tensor_parallel

                    # Get TP ports
                    next_open_port = cls._used_tp_ports[-1] + 1
                    tensor_parallel_ports = list(
                        range(next_open_port,
                              next_open_port + dep_conf.tensor_parallel))
                    cls._used_tp_ports.extend(tensor_parallel_ports)

                    # Get torch dist port
                    torch_dist_port = cls._used_dist_ports[-1] + 1
                    cls._used_dist_ports.extend([torch_dist_port])

                    dep_conf.replica_configs.append({
                        "gpu_indices": gpu_indices,
                        "tensor_parallel_ports": tensor_parallel_ports,
                        "torch_dist_port": torch_dist_port,
                    })
        return fields
