import mii

mii_configs = {'tensor_parallel': 1}

# gpt2
name = "microsoft/DialoGPT-small"

print(f"Deploying {name}...")

mii.deploy(
    task_name='conversational',
    model_name=name,
    deployment_type=mii.DeploymentType.LOCAL,
    deployment_name=name + "_deployment",
    local_model_path=".cache/models/" + name,
)
