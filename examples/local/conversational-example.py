import mii
from mii.constants import PORT_NUMBER_KEY

mii_configs = mii.constants.MII_CONFIGS_DEFAULT
mii_configs[mii.constants.TENSOR_PARALLEL_KEY] = 2

# gpt2
name = "microsoft/DialoGPT-large"

print(f"Deploying {name}...")

mii.deploy('conversational',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path=".cache/models/" + name,
           enable_deepspeed=True)
