# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from .api import client, serve, pipeline

from .legacy import MIIServer, MIIClient, mii_query_handle, deploy, terminate, DeploymentType, TaskType, aml_output_path, MIIConfig, ModelConfig, get_supported_models

__version__ = "0.0.0"
try:
    from .version import __version__
except ImportError:
    pass
