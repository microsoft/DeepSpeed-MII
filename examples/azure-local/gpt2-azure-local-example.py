import os
import mii

from azureml.core import Workspace

# TODO: align these environment vars w. azure naming style
ws = Workspace(workspace_name=os.environ["AZ_WORKSPACE"],
               subscription_id=os.environ["AZ_SUB_ID"],
               resource_group=os.environ["AZ_RESOURCE_GROUP"])

mii.deploy(task_name="text-generation",
           model_name="gpt2",
           aml_model_tags={'my_tag': 'first_deployment'},
           deployment_type=mii.DeploymentType.AML_LOCAL,
           aml_workspace=ws,
           aml_deployment_name="my-gpt2-service",
           local_model_path=None,
           force_register_model=False)
