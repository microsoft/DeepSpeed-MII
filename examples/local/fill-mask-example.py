import mii

# roberta
name = "roberta-base"
name = "bert-base-cased"

print(f"Deploying {name}...")

mii.deploy(task='fill-mask', model=name, deployment_name=name + "_deployment")
