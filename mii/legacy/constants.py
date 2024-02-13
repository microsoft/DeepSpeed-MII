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
    TEXT_CLASSIFICATION = "text-classification"
    QUESTION_ANSWERING = "question-answering"
    FILL_MASK = "fill-mask"
    TOKEN_CLASSIFICATION = "token-classification"
    CONVERSATIONAL = "conversational"
    TEXT2IMG = "text-to-image"
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    INPAINTING = "text-to-image-inpainting"


class ModelProvider(str, Enum):
    HUGGING_FACE = "hugging-face"
    ELEUTHER_AI = "eleuther-ai"
    DIFFUSERS = "diffusers"


SUPPORTED_MODEL_TYPES = {
    'roberta': ModelProvider.HUGGING_FACE,
    'xlm-roberta': ModelProvider.HUGGING_FACE,
    'gpt2': ModelProvider.HUGGING_FACE,
    'distilbert': ModelProvider.HUGGING_FACE,
    'bert': ModelProvider.HUGGING_FACE,
    'gpt_neo': ModelProvider.HUGGING_FACE,
    'gptj': ModelProvider.HUGGING_FACE,
    'opt': ModelProvider.HUGGING_FACE,
    'bloom': ModelProvider.HUGGING_FACE,
    'gpt-neox': ModelProvider.ELEUTHER_AI,
    'stable-diffusion': ModelProvider.DIFFUSERS,
    'llama': ModelProvider.HUGGING_FACE,
    'clip': ModelProvider.HUGGING_FACE
}

REQUIRED_KEYS_PER_TASK = {
    TaskType.TEXT_GENERATION: ["query"],
    TaskType.TEXT_CLASSIFICATION: ["query"],
    TaskType.QUESTION_ANSWERING: ["context",
                                  "question"],
    TaskType.FILL_MASK: ["query"],
    TaskType.TOKEN_CLASSIFICATION: ["query"],
    TaskType.CONVERSATIONAL: [
        "text",
        "conversation_id",
        "past_user_inputs",
        "generated_responses",
    ],
    TaskType.TEXT2IMG: ["prompt"],
    TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION: ["image",
                                              "candidate_labels"],
    TaskType.INPAINTING: [
        "prompt",
        "image",
        "mask_image",
    ]
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
CREATE_SESSION_METHOD = "CreateSession"
DESTROY_SESSION_METHOD = "DestroySession"

LB_MAX_WORKER_THREADS = 32

SERVER_SHUTDOWN_TIMEOUT = 10

RESTFUL_GATEWAY_SHUTDOWN_TIMEOUT = 1
RESTFUL_API_PATH = "mii"
