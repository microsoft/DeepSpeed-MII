import mii

mii_config = {"dtype": "fp16"}

name = "distilgpt2"
name = "gpt2-xl"
mii.deploy('text-generation',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path=".cache/models/" + name,
           mii_configs=mii_configs,
           enable_deepspeed=False,
           enable_zero=True)
