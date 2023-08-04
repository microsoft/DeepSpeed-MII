# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from typing import Union, List
from enum import Enum
from pydantic import validator, root_validator

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from deepspeed.runtime.config import DTypeEnum
from deepspeed.launcher.runner import DLTS_HOSTFILE


class DeploymentType(Enum):
    LOCAL = "local"
    AML = "aml"
    NON_PERSISTENT = "non-persistent"


class TaskType(Enum):
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    QUESTION_ANSWERING = "question-answering"
    FILL_MASK = "fill-mask"
    TOKEN_CLASSIFICATION = "token-classification"
    CONVERSATIONAL = "conversational"
    TEXT2IMG = "text-to-image"


class ReplicaConfig(BaseModel):
    hostname: str = ""
    tensor_parallel_ports: List[int] = []
    torch_dist_port: int = None
    gpu_indices: List[int] = []


class DeploymentConfig(DeepSpeedConfigModel):
    deployment_name: str
    model: str
    task: TaskType
    tensor_parallel: int = 1
    dtype: DtypeEnum = torch.float32
    meta_tensor: bool = False
    load_with_sys_mem: bool = False
    enable_cuda_graph: bool = False
    checkpoint_dict: Optional[Dict[str, Any]] = None
    deploy_rank: Optional[List[int]] = None
    torch_dist_port: int = 29500
    replace_with_kernel_inject: bool = True
    profile_model_time: bool = False
    skip_model_check: bool = False
    max_tokens: int = 1024
    trust_remote_code: bool = False
    enable_deepspeed: bool = True
    enable_zero: bool = False
    ds_config: Dict[str, Any] = {}
    model_path: Optional[str] = ""
    replica_num: int = 1
    replica_configs: List[ReplicaConfig] = []

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

    @validator("deploy_rank", pre=True)
    def deploy_rank_to_list(cls, field_value, values):
        if not isinstance(field_value, list):
            field_value = [field_value]
        return field_value

    @root_validator
    def deploy_rank_valid(cls, values):
        tensor_parallel = values.get("tensor_parallel")
        deploy_rank = values.get("deploy_rank")

        # if deploy rank is not given, default to align with TP value
        if deploy_rank is None:
            deploy_rank = list(range(tensor_parallel))

        # number of ranks provided must be equal to TP size, DP is handled outside MII currently
        assert tensor_parallel == len(deploy_rank), \
            f"{len(deploy_rank)} rank(s) provided in 'deploy_rank' does not align with tensor_parallel size of {tensor_parallel}"

        values["deploy_rank"] = deploy_rank
        return values

    @root_validator
    def set_model_path(cls, values):
        if values.get("model_path") is None:
            deployment_type = values.get("deployment_type")
            if deployment_type == DeploymentType.LOCAL:
                model_path = MII_MODEL_PATH_DEFAULT
            if deployment_tpye == DeploymentType.AML:
                model_path = "model"
            values["model_path"] = model_path
        return values

    @root_validator
    def validate_model_and_task(cls, values):
        task = values.get("task")
        model = values.get("model")
        if not values.get("skip_model_check"):
            mii.utils.check_if_task_and_model_is_valid(task, model)
            if values.get("ds_optimize"):
                mii.utils.check_if_task_and_model_is_supported(task, model)
        return values

    @root_validator
    def meta_tensor_or_sys_mem(cls, values):
        if values.get("meta_tensor") and values.get("load_with_sys_mem"):
            raise ValueError(
                "`meta_tensor` and `load_with_sys_mem` cannot be active at the same time."
            )
        return values

    @root_validator
    def zero_dtype_valid(cls, values):
        if values.get("enable_zero"):
            if values.get("ds_config").get("fp16", {}).get("enabled", False):
                assert (
                    values.get("dtype") == DtypeEnum.float16
                ), "ZeRO FP16 enabled, `dtype` must be set to `torch.float16`"
            else:
                assert (
                    values.get("dtype") == DtypeEnum.float32
                ), "ZeRO FP16 disabled, `dtype` must be set to `torch.float32`"
        return values

    @root_validator
    def deepspeed_or_zero(cls, values):
        assert not (
            values.get("enable_deepspeed") and values.get("enable_zero")
        ), "DeepSpeed and ZeRO cannot both be enabled, select only one"
        return values


class MIIConfig(DeepSpeedConfigModel):
    hf_auth_token: str = None
    port_number: int = 50050
    enable_restful_api: bool = False
    restful_api_port: int = 51080
    hostfile: str = DLTS_HOSTFILE
    deployment_type: DeploymentType = DeploymentType.LOCAL
    version: int = 1

    @root_validator
    def AML_name_valid(cls, fields):
        if fields.get("deployment_type") == DeploymentType.AML:
            allowed_chars = set(string.ascii_lowercase + string.ascii_uppercaes +
                                string.digits + "-")
            assert (
                set(fields.get("deployment_config").deployment_name) <= allowed_chars
            ), "AML deployment names can only contain a-z, A-Z, 0-9, and '-'."
        return fields

    @root_validator
    def generate_replica_configs(cls, values):
        replica_configs = values.get("deployment_config").replica_configs
        num_replicas = values.get("deployment_config").num_replicas
        if replica_configs:
            assert len(replica_confgs) == num_replicas
            return values

        hostfile = values.get("hostfile")
        port_number = values.get("port_number")
        torch_dist_port = values.get("deployment_config").torch_dist_port
        tensor_parallel = values.get("deployment_config").tensor_parallel
        replica_num = values.get("deployment_config").replica_num
        replica_pool = _allocate_processes(hostfile, tensor_parallel, replica_num)
        replica_configs = []
        for i, (hostname, gpu_indices) in enumerate(replica_pool):
            # Reserver port for a LB proxy when replication is enabled
            port_offset = 1
            base_port = port_number + i * tensor_parallel + port_offset
            tensor_parallel_ports = list(range(base_port, base_port + tensor_parallel))
            replica_torch_dist_port = torch_dist_port + i
            replica_configs.append(
                ReplicaConfig(hostname=hostname,
                              tensor_parallel_ports=tensor_parallel_ports,
                              torch_dist_port=replica_torch_dist_port,
                              gpu_indices=gpu_indices))

        values.get("deployment_config").replica_configs = replica_configs
        return values


def _allocate_processes(hostfile_path, tensor_parallel, num_replicas):
    resource_pool = fetch_hostfile(hostfile_path)
    assert resource_pool is not None and len(
        resource_pool) > 0, f'No hosts found in {hostfile_path}'

    replica_pool = []
    allocated_num = 0
    for host, slots in resource_pool.items():
        available_on_host = slots
        while available_on_host >= tensor_parallel:
            if allocated_num >= num_replicas:
                break
            if slots < tensor_parallel:
                raise ValueError(
                    f'Host {host} has {slots} slot(s), but {tensor_parallel} slot(s) are required'
                )

            allocated_num_on_host = slots - available_on_host
            replica_pool.append(
                (host,
                 [
                     i for i in range(allocated_num_on_host,
                                      allocated_num_on_host + tensor_parallel)
                 ]))
            allocated_num += 1

            available_on_host -= tensor_parallel

    if allocated_num < num_replicas:
        raise ValueError(
            f'No sufficient GPUs for {num_replicas} replica(s), only {allocated_num} replica(s) can be deployed'
        )

    return replica_pool
