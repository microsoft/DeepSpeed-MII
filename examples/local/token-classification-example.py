import mii

# roberta
name = "Jean-Baptiste/roberta-large-ner-english"

print(f"Deploying {name}...")

mii.deploy('token-classification',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path=".cache/models/" + name)
