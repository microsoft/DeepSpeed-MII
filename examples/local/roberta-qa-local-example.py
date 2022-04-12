import mii

name="distilroberta-base"
mii.deploy("question-answering",
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name+"-qa-deployment",
           local_model_path=".cache/models/"+name)
