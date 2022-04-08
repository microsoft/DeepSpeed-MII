import mii

mii.deploy("question-answering",
           "deepset/roberta-large-squad2",
           mii.DeploymentType.LOCAL,
           deployment_name="roberta-qa-deployment",
           local_model_path=".cache/models/roberta-large")
