import mii

mii_configs = {'dtype': 'fp16'}

name = "facebook/opt-1.3b"

ds_config = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
        },
    },
    "train_micro_batch_size_per_gpu": 1,
}

mii.deploy('text-generation',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path=".cache/models/" + name,
           mii_configs=mii_configs,
           enable_deepspeed=False,
           enable_zero=True,
           ds_config=ds_config)
