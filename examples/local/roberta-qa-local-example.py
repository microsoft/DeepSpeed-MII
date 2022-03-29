import mii

mii.deploy("question-answering",
           "deepset/roberta-large-squad2",
           mii.DeploymentType.LOCAL,
           local_model_path=".cache/models/gpt2")
