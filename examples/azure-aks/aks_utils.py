from azureml.core.compute import ComputeTarget, AksCompute
from azureml.exceptions import ComputeTargetException
import mii


def get_aks_target(aks_name, aml_workspace, aks_provision_config=None):
    # Check to see if the cluster already exists
    try:
        aks_target = ComputeTarget(workspace=aml_workspace, name=aks_name)
        mii.utils.logger.info('Found existing compute target')

    except ComputeTargetException:
        mii.utils.logger.warn("Cannot find the AKS cluster. Attempting to create one.")

        # Provision AKS cluster with GPU machine
        if aks_provision_config is None:
            mii.utils.logger.warn(
                f"aks_provision_config is {aks_provision_config}, attempting to use a default settings "
            )
            prov_config = AksCompute.provisioning_configuration(
                vm_size="Standard_NC6_Promo")
        else:
            prov_config = aks_provision_config

        # Create the cluster
        aks_target = ComputeTarget.create(workspace=aml_workspace,
                                          name=aks_name,
                                          provisioning_configuration=prov_config)
        aks_target.wait_for_completion(show_output=True)
        mii.utils.logger.info(f"AKS cluster {aks_name} successfully created")

    return aks_target
