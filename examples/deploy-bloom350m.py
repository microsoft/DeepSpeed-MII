import mii

mii_config = {"tensor_parallel": 1, "dtype": "fp16"}

mii.deploy(task='text-generation',
           model="gpt2",
           deployment_name="gpt2_deployment",
           mii_config=mii_config)
