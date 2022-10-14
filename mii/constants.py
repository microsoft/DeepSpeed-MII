import enum


#TODO naming..
class DeploymentType(enum.Enum):
    LOCAL = 1
    AML = 2


MII_CONFIGS_KEY = 'mii_configs'


class Tasks(enum.Enum):
    TEXT_GENERATION = 1
    TEXT_CLASSIFICATION = 2
    QUESTION_ANSWERING = 3
    FILL_MASK = 4
    TOKEN_CLASSIFICATION = 5
    CONVERSATIONAL = 6
    TEXT2IMG = 7


TEXT_GENERATION_NAME = 'text-generation'
TEXT_CLASSIFICATION_NAME = 'text-classification'
QUESTION_ANSWERING_NAME = 'question-answering'
FILL_MASK_NAME = 'fill-mask'
TOKEN_CLASSIFICATION_NAME = 'token-classification'
CONVERSATIONAL_NAME = 'conversational'
TEXT2IMG_NAME = "text-to-image"


class ModelProvider(enum.Enum):
    HUGGING_FACE = 1
    ELEUTHER_AI = 2
    HUGGING_FACE_LLM = 3
    DIFFUSERS = 4


MODEL_PROVIDER_NAME_HF = "hugging-face"
MODEL_PROVIDER_NAME_EA = "eleuther-ai"
MODEL_PROVIDER_NAME_HF_LLM = "hugging-face-llm"
MODEL_PROVIDER_NAME_DIFFUSERS = "diffusers"

MODEL_PROVIDER_MAP = {
    MODEL_PROVIDER_NAME_HF: ModelProvider.HUGGING_FACE,
    MODEL_PROVIDER_NAME_EA: ModelProvider.ELEUTHER_AI,
    MODEL_PROVIDER_NAME_HF_LLM: ModelProvider.HUGGING_FACE_LLM,
    MODEL_PROVIDER_NAME_DIFFUSERS: ModelProvider.DIFFUSERS
}

SUPPORTED_MODEL_TYPES = {
    'roberta': ModelProvider.HUGGING_FACE,
    'gpt2': ModelProvider.HUGGING_FACE,
    'bert': ModelProvider.HUGGING_FACE,
    'gpt_neo': ModelProvider.HUGGING_FACE,
    'gptj': ModelProvider.HUGGING_FACE,
    'opt': ModelProvider.HUGGING_FACE,
    'gpt-neox': ModelProvider.ELEUTHER_AI,
    'bloom': ModelProvider.HUGGING_FACE_LLM,
    'stable-diffusion': ModelProvider.DIFFUSERS
}

SUPPORTED_TASKS = [
    TEXT_GENERATION_NAME,
    TEXT_CLASSIFICATION_NAME,
    QUESTION_ANSWERING_NAME,
    FILL_MASK_NAME,
    TOKEN_CLASSIFICATION_NAME,
    CONVERSATIONAL_NAME,
    TEXT2IMG_NAME
]

REQUIRED_KEYS_PER_TASK = {
    TEXT_GENERATION_NAME: ["query"],
    TEXT_CLASSIFICATION_NAME: ["query"],
    QUESTION_ANSWERING_NAME: ["context",
                              "question"],
    FILL_MASK_NAME: ["query"],
    TOKEN_CLASSIFICATION_NAME: ["query"],
    CONVERSATIONAL_NAME:
    ['text',
     'conversation_id',
     'past_user_inputs',
     'generated_responses'],
    TEXT2IMG_NAME: ["query"]
}

MODEL_NAME_KEY = 'model_name'
TASK_NAME_KEY = 'task_name'
MODEL_PATH_KEY = 'model_path'

ENABLE_DEEPSPEED_KEY = 'ds_optimize'
ENABLE_DEEPSPEED_ZERO_KEY = 'ds_zero'
DEEPSPEED_CONFIG_KEY = 'ds_config'
CHECKPOINT_KEY = "checkpoint"

MII_CACHE_PATH = "MII_CACHE_PATH"
MII_CACHE_PATH_DEFAULT = "/tmp/mii_cache"

MII_DEBUG_MODE = "MII_DEBUG_MODE"
MII_DEBUG_MODE_DEFAULT = "0"

MII_DEBUG_DEPLOY_KEY = "MII_DEBUG_DEPLOY_KEY"

MII_DEBUG_BRANCH = "MII_DEBUG_BRANCH"
MII_DEBUG_BRANCH_DEFAULT = "main"

MII_MODEL_PATH_DEFAULT = "/tmp/mii_models"

GRPC_MAX_MSG_SIZE = 2**30  # 1GB
