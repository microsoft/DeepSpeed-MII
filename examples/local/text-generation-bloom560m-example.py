import mii

rank = 0

mii_config = {
    "tensor_parallel": 1,
    "dtype": "fp16",
    "deploy_rank": rank,
    "port_number": f"{50050 + rank}"
}

mii.deploy(task='text-generation',
           model="EleutherAI/gpt-j-6B",
           deployment_name=f"gptj_{rank}",
           mii_config=mii_config)
