import mii
from transformers import AutoConfig

name = "distilgpt2"
name = "gpt2-xl"

config = AutoConfig.from_pretrained(name)
model_hidden_size = config.n_embd

ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/mnt/nvme0/offload",
        },
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.1 * model_hidden_size * model_hidden_size,
        "stage3_max_live_parameters": 1e8,
        "stage3_max_reuse_distance": 1e8,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "train_micro_batch_size_per_gpu": 1,
}

# or give a path to a config file
# ds_config = "./tmp_config.json"

mii.deploy('text-generation',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path=".cache/models/" + name,
           enable_deepspeed=False,
           enable_zero=True,
           ds_config=ds_config)
