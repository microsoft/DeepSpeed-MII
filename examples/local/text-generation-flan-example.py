import mii

mii_configs = {"tensor_parallel": 1, "dtype": "fp16"}
mii.deploy(task='text2text-generation',
           model="google/flan-t5-small",
           deployment_name="flan_deployment",
           mii_config=mii_configs)
