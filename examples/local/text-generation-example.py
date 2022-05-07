import mii
from transformers import AutoConfig

mii_config = {"dtype": "fp16"}

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
            "pin_memory": True,
            "buffer_count": 6,
            "buffer_size": 1e9,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.1 * model_hidden_size * model_hidden_size,
        "stage3_max_live_parameters": 1e8,
        "stage3_max_reuse_distance": 1e8,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
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
