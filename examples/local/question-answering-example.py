import mii

name = "deepset/roberta-large-squad2"
mii.deploy("question-answering",
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "-qa-deployment",
           local_model_path=".cache/models/" + name)
