# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import string
from typing import List, Optional, Union, Dict, Any

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from deepspeed.launcher.runner import DLTS_HOSTFILE, fetch_hostfile
from deepspeed.inference import RaggedInferenceEngineConfig

from mii.constants import DeploymentType, TaskType, ModelProvider
from mii.errors import DeploymentNotFoundError
from mii.modeling.tokenizers import MIITokenizerWrapper
from mii.pydantic_v1 import Field, root_validator
from mii.utils import generate_deployment_name, get_default_task, import_score_file


class ReplicaConfig(DeepSpeedConfigModel):
    hostname: str = ""
    tensor_parallel_ports: List[int] = []
    torch_dist_port: int = None
    gpu_indices: List[int] = []
    zmq_port: int = None


class ModelConfig(DeepSpeedConfigModel):
    model_name_or_path: str
    """
    Model name or path of the model to HuggingFace model to be deployed.
    """

    tokenizer: Optional[Union[str, MIITokenizerWrapper]] = None
    """
    Tokenizer wrapped with `MIITokenizerWrapper`, name or path of the
    HuggingFace tokenizer to be used.
    """

    task: Optional[TaskType] = TaskType.TEXT_GENERATION
    """
    Name of the task to be performed by the model.
    """

    tensor_parallel: int = int(os.getenv("WORLD_SIZE", "1"))
    """
    Tensor parallelism to use for a model (i.e., how many GPUs to shard a model
    across). This defaults to the `WORLD_SIZE` environment variable, or a value
    of 1 if that variable is not set. This value is also propagated to the
    `inference_engine_config`.
    """

    inference_engine_config: RaggedInferenceEngineConfig = {}
    """
    DeepSpeed inference engine config. This is automatically generated, but you
    can provide a set of custom configs.
    """

    torch_dist_port: int = 29500
    """
    Torch distributed port to be used. This also serves as a base port when
    multiple replicas are deployed. For example, if there are 2 replicas, the
    first will use port 29500 and the second will use port 29600.
    """

    zmq_port_number: int = 25555
    """
    Port number to use for the ZMQ communication (for broadcasting requests and
    responses among all ranks in ragged batching).
    """

    replica_num: int = Field(1, gt=0)
    """
    Number of model replicas. Enables easy data parallelism.
    """

    replica_configs: List[ReplicaConfig] = []
    """
    Configuration details for each replica. This will be automatically
    generated, but you can provide a set of custom configs.
    """

    max_length: Optional[int] = None
    """
    The maximum number of tokens DeepSpeed-Inference can work with, including
    the input and output tokens.
    """

    all_rank_output: bool = False
    """
    Weather to return output on all ranks for `mii.pipeline`. Default behavior
    is to only return on rank 0.
    """

    sync_debug: bool = False
    """
    Inserts additional synchronization points for debugging purposes.
    """

    profile_model_time: bool = False
    """
    Log performance information about model inference with very little overhead.
    """
    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.HUGGING_FACE

    @root_validator
    def auto_fill_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("tokenizer"):
            values["tokenizer"] = values.get("model_name_or_path")
        if not values.get("task"):
            values["task"] = get_default_task(values.get("model_name_or_path"))
        return values

    @root_validator
    def propagate_tp_size(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        tensor_parallel = values.get("tensor_parallel")
        values.get("inference_engine_config").tensor_parallel.tp_size = tensor_parallel
        return values

    @root_validator
    def check_replica_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        num_replica_config = len(values.get("replica_configs"))
        if num_replica_config > 0:
            assert num_replica_config == values.get("replica_num"), "Number of replica configs must match replica_num"
        return values


class MIIConfig(DeepSpeedConfigModel):
    deployment_name: str = ""
    """
    Name of the deployment. Used as an identifier for obtaining a inference
    server client and posting queries. Automatically generated if it is not provided.
    """

    deployment_type: DeploymentType = DeploymentType.LOCAL
    """
    One of the `enum mii.DeploymentTypes:`
    * `LOCAL` uses a grpc server to create a local deployment.
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
    def AML_name_valid(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("deployment_type") == DeploymentType.AML:
            allowed_chars = set(string.ascii_lowercase + string.ascii_uppercase +
                                string.digits + "-")
            assert (
                set(values.get("deployment_name")) <= allowed_chars
            ), "AML deployment names can only contain a-z, A-Z, 0-9, and '-'."
        return values

    @root_validator(skip_on_failure=True)
    def check_deployment_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        deployment_name = values.get("deployment_name")
        if not deployment_name:
            model_name_or_path = values.get("model_config").model_name_or_path
            deployment_name = generate_deployment_name(
                model_name_or_path=model_name_or_path)
            values["deployment_name"] = deployment_name
        return values

    def generate_replica_configs(self) -> None:
        if self.model_config.replica_configs:
            return
        # TODO: refactor this function
        hostfile = self.hostfile
        port_number = self.port_number
        torch_dist_port = self.model_config.torch_dist_port
        tensor_parallel = self.model_config.tensor_parallel
        zmq_port = self.model_config.zmq_port_number
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
                    zmq_port=zmq_port + i,
                ))

        self.model_config.replica_configs = replica_configs


def _allocate_processes(hostfile_path: str, tensor_parallel: int, replica_num: int):
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


def get_mii_config(model_or_deployment_name: str) -> MIIConfig:
    try:
        deployment_name = model_or_deployment_name
        mii_config = import_score_file(deployment_name, DeploymentType.LOCAL).mii_config
    except:
        try:
            deployment_name = generate_deployment_name(
                model_name_or_path=model_or_deployment_name)
            mii_config = import_score_file(deployment_name,
                                           DeploymentType.LOCAL).mii_config
        except:
            raise DeploymentNotFoundError(
                f"Could not find a deployment named {model_or_deployment_name} or {deployment_name}"
            )
    return MIIConfig(**mii_config)
