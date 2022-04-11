import os
import mii

from azureml.core import Workspace
import aks_utils
from azureml.core.webservice import AksWebservice

# TODO: align these environment vars w. azure naming style
ws = Workspace(workspace_name=os.environ["AZ_WORKSPACE"],
               subscription_id=os.environ["AZ_SUB_ID"],
               resource_group=os.environ["AZ_RESOURCE_GROUP"])

#aks deployment config. Use your own
gpu_aks_config = AksWebservice.deploy_configuration(autoscale_enabled=False,
                                                    num_replicas=1,
                                                    cpu_cores=2,
                                                    gpu_cores=2,
                                                    memory_gb=10)
mii.deploy(task_name="text-generation",
           model_name="gpt2",
           deployment_type=mii.DeploymentType.AML_ON_AKS,
           deployment_name="my-gpt2-service",
           aml_workspace=ws,
           aml_model_tags={'my_tag': 'first_deployment'},
           aks_target=aks_utils.get_aks_target("V100-node",
                                               ws),
           aks_deploy_config=gpu_aks_config)
