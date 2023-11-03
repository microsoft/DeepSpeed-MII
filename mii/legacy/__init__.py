# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import grpc
from .server import MIIServer
from .client import MIIClient, mii_query_handle
from .deployment import deploy
from .terminate import terminate
from .constants import DeploymentType, TaskType
from .aml_related.utils import aml_output_path
from .config import MIIConfig, ModelConfig
from .utils import get_supported_models
from .grpc_related.proto import legacymodelresponse_pb2_grpc as modelresponse_pb2_grpc

__version__ = "0.0.0"
non_persistent_models = {}
try:
    from .version import __version__
except ImportError:
    pass
