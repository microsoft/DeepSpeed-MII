import mii

# roberta
name = "roberta-base"
name = "roberta-large"

# name = "bert-base-uncased"

# name = "bert-base-uncased"

print(f"Deploying {name}...")

mii.deploy('fill-mask',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path=".cache/models/" + name,
           enable_deepspeed=True)
