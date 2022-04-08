import enum


#TODO naming..
class DeploymentType(enum.Enum):
    LOCAL = 1
    #expose GPUs
    AML_LOCAL = 2
    AML_ON_AKS = 3


class Parallelism(enum.Enum):
    Tensor = 1
    Pipeline = 2
    Expert =3

MII_PARALLELISM_DEFAULT = {Parallelism.Tensor: 1, Parallelism.Pipeline: 1, Parallelism.Expert: 1}


class Tasks(enum.Enum):
    TEXT_GENERATION = 1
    TEXT_CLASSIFICATION = 2
    QUESTION_ANSWERING = 3


TEXT_GENERATION_NAME = 'text-generation'
TEXT_CLASSIFICATION_NAME = 'text-classification'
QUESTION_ANSWERING_NAME = 'question-answering'


MODEL_NAME_KEY = 'model_name'
TASK_NAME_KEY = 'task_name'
PARALLELISM_KEY = 'parallelism'
ENABLE_DEEPSPEED_KEY = 'ds_optimize'

MII_CACHE_PATH = "MII_CACHE_PATH"
MII_CACHE_PATH_DEFAULT = "/tmp/mii_cache"

MII_DEBUG_MODE = "MII_DEBUG_MODE"
MII_DEBUG_MODE_DEFAULT = "0"

MII_DEBUG_DEPLOY_KEY = "MII_DEBUG_DEPLOY_KEY"

MII_DEBUG_BRANCH = "MII_DEBUG_BRANCH"
MII_DEBUG_BRANCH_DEFAULT = "main"


