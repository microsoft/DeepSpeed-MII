import mii

mii_configs = {'tensor_parallel': 1, 'port_number': 50050}

name = "deepset/roberta-large-squad2"
mii.deploy("question-answering",
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path=".cache/models/" + name,
           mii_configs=mii_configs,
           enable_deepspeed=True)

mii_configs['port_number'] = 50051

mii.deploy("question-answering",
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "-qa-deployment2",
           local_model_path=".cache/models/" + name,
           mii_configs=mii_configs,
           enable_deepspeed=False)
