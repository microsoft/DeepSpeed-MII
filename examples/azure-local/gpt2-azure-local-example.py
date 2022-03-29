import os
import mii

from azureml.core import Workspace
from transformers import pipeline

# TODO: if local model path not given, do this code on mii side
os.environ["TRANSFORMERS_CACHE"] = "/tmp/mii_cache/gpt2"
generator = pipeline('text-generation', model='gpt2')

# TODO: align these environment vars w. azure naming style
ws = Workspace(workspace_name=os.environ["AZ_WORKSPACE"],
               subscription_id=os.environ["AZ_SUB_ID"],
               resource_group=os.environ["AZ_RESOURCE_GROUP"])

mii.deploy(task_name="text-generation",
           model_name="gpt2",
           deployment_type=mii.DeploymentType.AML_LOCAL,
           aml_workspace=ws,
           aml_deployment_name="myservice",
           local_model_path="/tmp/mii_cache/gpt2")
