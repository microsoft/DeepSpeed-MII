import mii

mii_configs = {"tensor_parallel": 1, "dtype": "fp16"}
mii.deploy(task='text-generation',
           model="bigscience/bloom-350m",
           deployment_name="bloom350m_deployment",
           deployment_type=mii.constants.DeploymentType.AML,
           mii_config=mii_configs)
