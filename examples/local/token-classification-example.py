import mii

# roberta
name = "Jean-Baptiste/roberta-large-ner-english"

print(f"Deploying {name}...")

mii.deploy(task='token-classification', model=name, deployment_name=name + "_deployment")
