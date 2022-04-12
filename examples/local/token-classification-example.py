import mii

mii_configs = mii.constants.MII_CONFIGS_DEFAULT
mii_configs[mii.constants.TENSOR_PARALLEL_KEY] = 2

# roberta
name = "Jean-Baptiste/roberta-large-ner-english"

print(f"Deploying {name}...")

mii.deploy('token-classification',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name+"_deployment",
           local_model_path=".cache/models/"+name,
           mii_configs=mii_configs)
