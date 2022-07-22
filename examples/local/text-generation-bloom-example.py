import mii

mii_configs = {"dtype": "fp16", "tensor_parallel": 8}
name = "bigscience/bloom"

mii.deploy(task='text-generation',
           model=name,
           deployment_name=name + "_deployment",
           local_model_path="/data/bloom",
           mii_config=mii_configs)
