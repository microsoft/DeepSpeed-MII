# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
try:
    import grpc
    from .pipeline import pipeline
    from .server import serve
    from .client import MIIClient, client
except ImportError as e:
    print("Warning: DeepSpeed-FastGen could not be imported:")
    print(e)
    pass

from .legacy import MIIServer, mii_query_handle, deploy, terminate, DeploymentType, TaskType, aml_output_path, MIIConfig, ModelConfig, get_supported_models

__version__ = "0.0.0"
try:
    from .version import __version__
except ImportError:
    pass
