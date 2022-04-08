import mii

mii_configs = mii.constants.MII_CONFIGS_DEFAULT
mii_configs[mii.constants.TENSOR_PARALLEL_KEY] = 2 
mii.deploy('text-generation',
           "gpt2",
           mii.DeploymentType.LOCAL,
           deployment_name = "gpt2_deployment",
           local_model_path=".cache/models/gpt2")
