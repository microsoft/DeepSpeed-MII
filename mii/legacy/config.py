# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import os
import string
from typing import List, Optional, Dict, Any
import mii.legacy as mii
from .constants import DeploymentType, TaskType, ModelProvider, MII_MODEL_PATH_DEFAULT
from .pydantic_v1 import validator, root_validator, Field

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from deepspeed.inference.config import DtypeEnum
from deepspeed.launcher.runner import DLTS_HOSTFILE, fetch_hostfile


class ReplicaConfig(DeepSpeedConfigModel):
    hostname: str = ""
    tensor_parallel_ports: List[int] = []
    torch_dist_port: int = None
    gpu_indices: List[int] = []


class ModelConfig(DeepSpeedConfigModel):
    model: str
    """
    Name of a supported model for the task. Models in MII are sourced from
    multiple open-source projects such as Huggingface Transformer, FairSeq,
    EluetherAI etc. For the list of supported models for each task, please see
    here [TODO].
    """

    task: TaskType
    """
    Name of the machine learning task to be deployed.Currently MII supports the
    following list of tasks ``['text-generation', 'text-classification',
    'question-answering', 'fill-mask', 'token-classification',
    'conversational', 'text-to-image']``
    """

    dtype: DtypeEnum = DtypeEnum.fp32
    """
    Desired model data type, will convert model to this type.  Supported target
    types: `torch.half`, `torch.float`, `torch.int8` (for BLOOM models)
    """

    model_path: str = ""
    """
    In LOCAL deployments this is the local path where model checkpoints are
    available. In AML deployments this is an optional relative path with
    AZURE_MODEL_DIR for the deployment.
    """

    load_with_sys_mem: bool = False
    """
    Loads the model onto system memory instead of GPU memory. This can help
    avoid OOM errors when sharding a model across several GPUs because MII will
    try to load a full copy of each model onto each GPU initially.
    """

    meta_tensor: bool = False
    """
    Loads the initial HuggingFace model using Meta Tensors that use no memory.
    Can dramatically improve load time and reduce memory requirements on
    supported models. Supported for GPT-J, GPT-NeoX, OPT, and BLOOM when kernel
    injection is enabled. Supported for all models when kernel injection is
    disabled.
    """

    deploy_rank: Optional[List[int]] = None
    """
    GPU indices a model is deployed on. Note that CUDA_VISIBLE_DEVICES does not
    work with DeepSpeed-MII.
    """

    torch_dist_port: int = 29500
    """
    Torch distributed port.
    """

    replica_num: int = 1
    """
    Number of model replicas. Enables easy data parallelism.
    """

    replica_configs: List[ReplicaConfig] = []
    """
    Configuration details for each replica. This will be automatically
    generated, but you can provide a set of custom configs.
    """

    profile_model_time: bool = False
    """
    Enable profiling of model times (i.e., without communication overhead).
    """

    skip_model_check: bool = False
    """
    Skip validation that a model supports a given task.
    """

    hf_auth_token: Optional[str] = Field(
        None,
        deprecated=True,
        deprecated_msg=
        "Parameter will be removed. Please use the `pipeline_kwargs` field to pass kwargs to the HuggingFace pipeline creation.",
    )
    """
    HuggingFace authentication token for accessing models. Will be propagated
    to all ModelConfig if none are provided there.
    """

    trust_remote_code: bool = Field(
        False,
        deprecated=True,
        deprecated_msg=
        "Parameter will be removed. Please use the `pipeline_kwargs` field to pass kwargs to the HuggingFace pipeline creation.",
    )
    """
    HuggingFace `tranformer.pipeline` option for `trust_remote_code`.
    """

    pipeline_kwargs: Dict[str, Any] = {}
    """
    kwargs to be passed to HuggingFace's `transformer.pipeline`.
    """

    # TODO: Replace with DeepSpeedInferenceConfig
    enable_deepspeed: bool = True
    """
    Enable DeepSpeed-Inference.
    """

    enable_zero: bool = False
    """
    Enable Zero-Inference.
    """

    ds_config: Dict[str, Any] = {}
    """
    DeepSpeed config to use when Zero-Inference is enabled.
    """

    tensor_parallel: int = 1
    """
    Tensor parallelism to use for a model (i.e., how many GPUs to shard a model across).
    """

    enable_cuda_graph: bool = False
    """
    Enables CUDA Graph captures with DeepSpeed-Inference.
    """

    replace_with_kernel_inject: bool = True
    """
    Enable custom kernel injection with DeepSpeed-Inference.
    """

    checkpoint_dict: Optional[Dict[str, Any]] = None
    """
    DeepSpeed model checkpoint dict.
    """

    max_tokens: int = 1024
    """
    The maximum number of tokens DeepSpeed-Inference can work with, including
    the input and output tokens. Please consider increasing it to the required
    token-length required for your use-case.
    """
    class Config:
        json_encoders = {torch.dtype: lambda x: str(x)}

    @property
    def provider(self):
        return mii.utils.get_provider(self.model, self.task)

    @validator("checkpoint_dict")
    def checkpoint_dict_valid(cls, field_value, values):
        if field_value is None:
            return field_value
        for k in ["checkpoints", "version", "type", "base_dir"]:
            if not field_value.get(k, ""):
                raise ValueError(f"Missing key={k} in checkpoint_dict")
        return field_value

    @validator("deploy_rank", pre=True)
    def deploy_rank_to_list(cls, field_value, values):
        if field_value and not isinstance(field_value, list):
            field_value = [field_value]
        return field_value

    @root_validator
    def zero_or_meta(cls, values):
        if values.get("enable_zero"):
            assert not values.get(
                "meta_tensor"
            ), "ZeRO-Inference does not support meta tensors."
        return values

    @root_validator
    def bloom_model_valid(cls, values):
        if "bigscience/bloom" in values.get("model"):
            # TODO: SHould be albe to use DtypeEnum here
            assert values.get("dtype") in [
                torch.int8,
                torch.float16,
            ], "Bloom models only support fp16/int8."
            assert not values.get(
                "enable_cuda_graph"
            ), "Bloom models do not support CUDA Graph."
        return values

    @root_validator
    def deploy_rank_valid(cls, values):
        tensor_parallel = values.get("tensor_parallel")
        deploy_rank = values.get("deploy_rank")

        # if deploy rank is not given, default to align with TP value
        if deploy_rank is None:
            deploy_rank = list(range(tensor_parallel))

        # number of ranks provided must be equal to TP size, DP is handled outside MII currently
        assert tensor_parallel == len(
            deploy_rank
        ), f"{len(deploy_rank)} rank(s) provided in 'deploy_rank' does not align with tensor_parallel size of {tensor_parallel}"

        values["deploy_rank"] = deploy_rank
        return values

    @root_validator
    def set_model_path(cls, values):
        model_path = values.get("model_path")
        if not model_path:
            if values.get("deployment_type") == DeploymentType.AML:
                model_path = "model"
            else:
                model_path = MII_MODEL_PATH_DEFAULT
        aml_model_dir = os.environ.get("AZUREML_MODEL_DIR", None)
        if aml_model_dir and not model_path.startswith(aml_model_dir):
            assert os.path.isabs(
                aml_model_dir
            ), "AZUREML_MODEL_DIR={aml_model_dir} must be an absolute path."
            assert not os.path.isabs(
                model_path
            ), f"model_path={model_path} must be relative to append w/ AML path."
            model_path = os.path.join(aml_model_dir, model_path)

        values["model_path"] = model_path
        return values

    @root_validator
    def validate_model_and_task(cls, values):
        task = values.get("task")
        model = values.get("model")
        if not values.get("skip_model_check"):
            mii.utils.check_if_task_and_model_is_valid(task, model)
            if values.get("enable_deepspeed"):
                mii.utils.check_if_task_and_model_is_supported(task, model)
        # Skip any future checks
        values["skip_model_check"] = True
        return values

    @root_validator
    def meta_tensor_or_sys_mem(cls, values):
        if values.get("meta_tensor") and values.get("load_with_sys_mem"):
            raise ValueError(
                "`meta_tensor` and `load_with_sys_mem` cannot be active at the same time."
            )
        return values

    @root_validator
    def sys_mem_and_diffusers(cls, values):
        if values.get("load_with_sys_mem"):
            model = values.get("model")
            task = values.get("task")
            assert not (mii.utils.get_provider(model, task) == ModelProvider.DIFFUSERS), "`load_with_sys_mem` is not support with Stable Diffusion"
        return values

    @root_validator
    def zero_dtype_valid(cls, values):
        if values.get("enable_zero"):
            if values.get("ds_config").get("fp16", {}).get("enabled", False):
                # TODO: We should be able to use DtypeEnum instead of torch.float
                assert (
                    values.get("dtype") == torch.float16
                ), "ZeRO FP16 enabled, `dtype` must be set to `torch.float16`"
            else:
                assert (
                    values.get("dtype") == torch.float32
                ), "ZeRO FP16 disabled, `dtype` must be set to `torch.float32`"
        return values

    @root_validator
    def deepspeed_or_zero(cls, values):
        assert not (
            values.get("enable_deepspeed") and values.get("enable_zero")
        ), "DeepSpeed and ZeRO cannot both be enabled, select only one"
        return values


class MIIConfig(DeepSpeedConfigModel):
    deployment_name: str
    """
    Name of the deployment. Used as an identifier for obtaining a inference
    server client and posting queries.
    """

    deployment_type: DeploymentType = DeploymentType.LOCAL
    """
    One of the `enum mii.DeploymentTypes: [LOCAL]`.
    * `LOCAL` uses a grpc server to create a local deployment.
    * `NON_PERSISTENT` creates a local deployment that will end when the process exits.
    * `AML` will generate the assets necessary to deploy on AML resources.
    """

    model_config: ModelConfig
    """
    Configuration for the deployed model(s).
    """

    port_number: int = 50050
    """
    Port number to use for the load balancer process.
    """

    enable_restful_api: bool = False
    """
    Enables a RESTful API that can be queries with via http POST method.
    """

    restful_api_port: int = 51080
    """
    Port number to use for the RESTful API.
    """

    hostfile: str = DLTS_HOSTFILE
    """
    DeepSpeed hostfile. Will be autogenerated if None is provided.
    """

    # TODO: Place AML-related configs in subconfig
    version: int = 1
    """
    Version number to pass to AML deployments.
    """

    instance_type: str = "Standard_NC12s_v3"
    """
    AML instance type to use when create AML deployment assets.
    """
    @root_validator(skip_on_failure=True)
    def AML_name_valid(cls, values):
        if values.get("deployment_type") == DeploymentType.AML:
            allowed_chars = set(string.ascii_lowercase + string.ascii_uppercase +
                                string.digits + "-")
            assert (
                set(values.get("deployment_name")) <= allowed_chars
            ), "AML deployment names can only contain a-z, A-Z, 0-9, and '-'."
        return values

    def generate_replica_configs(self):
        # TODO: refactor this function
        hostfile = self.hostfile
        port_number = self.port_number
        torch_dist_port = self.model_config.torch_dist_port
        tensor_parallel = self.model_config.tensor_parallel
        replica_num = self.model_config.replica_num
        replica_pool = _allocate_processes(hostfile, tensor_parallel, replica_num)
        replica_configs = []
        for i, (hostname, gpu_indices) in enumerate(replica_pool):
            # Reserver port for a LB proxy when replication is enabled
            port_offset = 1
            base_port = port_number + i * tensor_parallel + port_offset
            tensor_parallel_ports = list(range(base_port, base_port + tensor_parallel))
            replica_torch_dist_port = torch_dist_port + (100 * i)
            replica_configs.append(
                ReplicaConfig(
                    hostname=hostname,
                    tensor_parallel_ports=tensor_parallel_ports,
                    torch_dist_port=replica_torch_dist_port,
                    gpu_indices=gpu_indices,
                ))

        self.model_config.replica_configs = replica_configs


def _allocate_processes(hostfile_path, tensor_parallel, replica_num):
    resource_pool = fetch_hostfile(hostfile_path)
    assert (
        resource_pool is not None and len(resource_pool) > 0
    ), f"No hosts found in {hostfile_path}"

    replica_pool = []
    allocated_num = 0
    for host, slots in resource_pool.items():
        available_on_host = slots
        while available_on_host >= tensor_parallel:
            if allocated_num >= replica_num:
                break
            if slots < tensor_parallel:
                raise ValueError(
                    f"Host {host} has {slots} slot(s), but {tensor_parallel} slot(s) are required"
                )

            allocated_num_on_host = slots - available_on_host
            replica_pool.append((
                host,
                [
                    i for i in range(
                        allocated_num_on_host,
                        allocated_num_on_host + tensor_parallel,
                    )
                ],
            ))
            allocated_num += 1

            available_on_host -= tensor_parallel

    if allocated_num < replica_num:
        raise ValueError(
            f"Not sufficient GPUs for {replica_num} replica(s), only {allocated_num} replica(s) can be deployed"
        )

    return replica_pool
