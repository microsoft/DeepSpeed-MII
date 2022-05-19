import mii

mii_config = {"tensor_parallel": 4, "port_number": 50050}

name = "gpt-neox"
mii.deploy('text-generation',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path="/data/20b",
           mii_configs=mii_config)
