import mii
from mii.constants import PORT_NUMBER_KEY

mii_configs = mii.constants.MII_CONFIGS_DEFAULT
mii_configs[mii.constants.TENSOR_PARALLEL_KEY] = 1
mii_configs[PORT_NUMBER_KEY] = 50050

name = "deepset/roberta-large-squad2"
mii.deploy("question-answering",
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path=".cache/models/" + name,
           mii_configs=mii_configs,
           enable_deepspeed=True)

mii_configs[PORT_NUMBER_KEY] = 50051

mii.deploy("question-answering",
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "-qa-deployment2",
           local_model_path=".cache/models/" + name,
           mii_configs=mii_configs,
           enable_deepspeed=False)
