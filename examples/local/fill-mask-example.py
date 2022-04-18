import mii

mii_configs = mii.constants.MII_CONFIGS_DEFAULT

# roberta
name = "roberta-base"
name = "vinai/bertweet-large"

# name = "bert-base-uncased"

print(f"Deploying {name}...")

mii.deploy('fill-mask',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path=".cache/models/" + name,
           mii_configs=mii_configs,
           enable_deepspeed=True)
