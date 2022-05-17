import mii

mii_configs = {'tensor_parallel': 2}

# gpt2
name = "microsoft/DialoGPT-small"

print(f"Deploying {name}...")

mii.deploy(
    'conversational',
    name,
    mii.DeploymentType.LOCAL,
    deployment_name=name + "_deployment",
    local_model_path=".cache/models/" + name,
)
