import mii

mii.deploy("text-generation",
           "gpt2",
           mii.DeploymentType.LOCAL,
           local_model_path=".cache/models/gpt2")
