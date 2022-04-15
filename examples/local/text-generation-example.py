import mii

mii_configs = mii.constants.MII_CONFIGS_DEFAULT

name = "distilgpt2"
mii.deploy('text-generation',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path=".cache/models/" + name,
           mii_configs=mii_configs)
