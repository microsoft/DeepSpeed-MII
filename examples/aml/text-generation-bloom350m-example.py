import mii

mii_configs = {
    "tensor_parallel": 1,
    "dtype": "fp16",
    "aml_model_path": "models/bloom-350m"
}
mii.deploy(task='text-generation',
           model="bigscience/bloom-350m",
           deployment_name="bloom350m_deployment",
           deployment_type=mii.constants.DeploymentType.AML,
           mii_config=mii_configs)
