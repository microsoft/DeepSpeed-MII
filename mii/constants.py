# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from enum import Enum


class DeploymentType(str, Enum):
    LOCAL = "local"
    AML = "aml"
    NON_PERSISTENT = "non-persistent"


class TaskType(str, Enum):
    TEXT_GENERATION = "text-generation"


class ModelProvider(str, Enum):
    HUGGING_FACE = "hugging-face"


class GenerationFinishReason(str, Enum):
    """ Reason for text-generation to stop. """

    STOP = "stop"
    """ Reached an EoS token. """

    LENGTH = "length"
    """ Reached ``max_length`` or ``max_new_tokens``. """

    NONE = "none"


SUPPORTED_MODEL_TYPES = {
    'opt': ModelProvider.HUGGING_FACE,
    'llama': ModelProvider.HUGGING_FACE
}

REQUIRED_KEYS_PER_TASK = {
    TaskType.TEXT_GENERATION: ["query"],
}

MII_CACHE_PATH = "MII_CACHE_PATH"
MII_CACHE_PATH_DEFAULT = "/tmp/mii_cache"

MII_HF_CACHE_EXPIRATION = "MII_HF_CACHE_EXPIRATION"
MII_HF_CACHE_EXPIRATION_DEFAULT = 60 * 60  # 1 hour

MII_DEBUG_MODE = "MII_DEBUG_MODE"
MII_DEBUG_MODE_DEFAULT = "0"

MII_DEBUG_DEPLOY_KEY = "MII_DEBUG_DEPLOY_KEY"

MII_DEBUG_BRANCH = "MII_DEBUG_BRANCH"
MII_DEBUG_BRANCH_DEFAULT = "main"

MII_MODEL_PATH_DEFAULT = "/tmp/mii_models"

GRPC_MAX_MSG_SIZE = 2**27  # ~100MB

TERMINATE_METHOD = "Terminate"

LB_MAX_WORKER_THREADS = 256

SERVER_SHUTDOWN_TIMEOUT = 10

RESTFUL_GATEWAY_SHUTDOWN_TIMEOUT = 1
RESTFUL_API_PATH = "mii"

STREAM_RESPONSE_QUEUE_TIMEOUT = 600
ZMQ_RECV_TIMEOUT = 5 * 1000
