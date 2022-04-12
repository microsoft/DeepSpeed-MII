import mii

mii_configs = mii.constants.MII_CONFIGS_DEFAULT
mii_configs[mii.constants.TENSOR_PARALLEL_KEY] = 2

# gpt2
name = "microsoft/DialogRPT-human-vs-rand"

# roberta
name = "roberta-large-mnli"

print(f"Deploying {name}...")

mii.deploy('text-classification',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name+"_deployment",
           local_model_path=".cache/models/"+name,
           mii_configs=mii_configs)
