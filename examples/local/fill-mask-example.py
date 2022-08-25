import mii

# roberta
name = "roberta-base"
name = "roberta-large"

# name = "bert-base-uncased"

# name = "bert-base-uncased"

print(f"Deploying {name}...")

mii.deploy(task='fill-mask', model=name, deployment_name=name + "_deployment")
