'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import sys
import os
import logging
import importlib

from mii.constants import MII_CACHE_PATH, MII_CACHE_PATH_DEFAULT, MII_DEBUG_MODE, \
    MII_DEBUG_MODE_DEFAULT, MII_DEBUG_DEPLOY_KEY, MII_DEBUG_BRANCH, MII_DEBUG_BRANCH_DEFAULT, \
    TEXT_GENERATION_NAME, TEXT_CLASSIFICATION_NAME, QUESTION_ANSWERING_NAME, TENSOR_PARALLEL_KEY, PORT_NUMBER_KEY

from mii.constants import Tasks


def validate_mii_configs(mii_configs):
    assert TENSOR_PARALLEL_KEY in mii_configs and PORT_NUMBER_KEY in mii_configs, "Missing configs {TENSOR_PARALLEL_KEY} and {PORT_NUMBER_KEY}"


def get_task_name(task):
    if task == Tasks.QUESTION_ANSWERING:
        return QUESTION_ANSWERING_NAME

    if task == Tasks.TEXT_GENERATION:
        return TEXT_GENERATION_NAME

    if task == Tasks.TEXT_CLASSIFICATION:
        return TEXT_CLASSIFICATION_NAME

    assert False, f"Unknown Task {task}"


def get_task(task_name):
    if task_name == QUESTION_ANSWERING_NAME:
        return Tasks.QUESTION_ANSWERING

    if task_name == TEXT_GENERATION_NAME:
        return Tasks.TEXT_GENERATION

    if task_name == TEXT_CLASSIFICATION_NAME:
        return Tasks.TEXT_CLASSIFICATION

    assert False, f"Unknown Task {task_name}"


#TODO read this from a file containing list of files supported for each task
def _get_supported_models_name(task):
    if task == Tasks.TEXT_GENERATION:
        supported_models = ['gpt2']

    elif task == Tasks.TEXT_CLASSIFICATION:
        supported_models = 'roberta-large-mnli'

    elif task == Tasks.QUESTION_ANSWERING:
        supported_models = ['deepset/roberta-large-squad2']
    else:
        assert False, f"{task} is not supported"

    return supported_models


def check_if_task_and_model_is_supported(task, model_name):
    supported = False
    supported_models = _get_supported_models_name(task)
    assert model_name in supported_models, f"{task} only supports {supported_models}"


def get_model_path():
    aml_model_dir = os.getenv('AZUREML_MODEL_DIR')
    if aml_model_dir is not None:
        return aml_model_dir

    mii_model_dir = os.getenv('MII_MODEL_DIR')

    if mii_model_dir is not None:
        return mii_model_dir

    #assert False, "MII_MODEL_DIR must be set if not running on AML. Current value is None"
    #TODO remove this and uncomment above line. Only doing this for debugging
    return "temp_model"


def is_aml():
    return os.getenv('AZUREML_MODEL_DIR') is not None


def set_model_path(model_path):
    os.environ['MII_MODEL_DIR'] = model_path


def mii_cache_path():
    cache_path = os.environ.get(MII_CACHE_PATH, MII_CACHE_PATH_DEFAULT)
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)
    return cache_path


def import_score_file(deployment_name):
    spec = importlib.util.spec_from_file_location(
        'score',
        os.path.join(mii_cache_path(),
                     deployment_name,
                     "score.py"))
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    return score


def generated_score_path(deployment_name):
    score_path = os.path.join(mii_cache_path(), deployment_name)
    if not os.path.isdir(score_path):
        os.makedirs(score_path)
    return os.path.join(score_path, "score.py")


def debug_score_preamble():
    preamble = ""
    debug_mode_enabled = int(os.environ.get(MII_DEBUG_MODE, MII_DEBUG_MODE_DEFAULT))
    if not debug_mode_enabled:
        return preamble

    deploy_key = os.environ.get(MII_DEBUG_DEPLOY_KEY)
    debug_branch = os.environ.get(MII_DEBUG_BRANCH, MII_DEBUG_BRANCH_DEFAULT)
    key_path = "/tmp/mii_deploy_key"

    preamble = f"""
import subprocess, os, sys
deploy_key = '''{deploy_key}'''
with open('{key_path}', 'w') as fd:
    fd.write(deploy_key)
    fd.write("\\n")
subprocess.run(['chmod', '600', '{key_path}'])
env = os.environ.copy()
env["GIT_SSH_COMMAND"]="ssh -i {key_path} -o StrictHostKeyChecking=no"
install_cmd = "-m pip install git+ssh://git@github.com/microsoft/DeepSpeed-MII.git@{debug_branch}"
subprocess.run([sys.executable] + install_cmd.split(" "), env=env)

"""
    return preamble


def setup_task():
    return get_model_path(), not is_aml(), is_aml()




log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerFactory:
    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger
        Args:
            name (str): name of the logger
            level: level of logger
        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] "
            "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_


logger = LoggerFactory.create_logger(name="MII", level=logging.INFO)
