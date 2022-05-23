# GPT-NeoX with MII
In this document, we provide the steps to setup MII for doing a local deployment of the [GPT-NeoX model](https://github.com/EleutherAI/gpt-neox).

## Setup Environment
We recommend using a conda environment or virtual environment for installing all dependencies:
```bash
# conda
conda create --name MII-GPT-NeoX
conda activate MII-GPT-NeoX
# python virtualenv
python3 -m venv MII-GPT-NeoX
source ./MII-GPT-NeoX/bin/activate
```
---
📌 **Note:** You should use Python3 <= 3.8. We recommend Python 3.8

---

## Install MII
```bash
git clone https://github.com/microsoft/DeepSpeed-MII.git
cd DeepSpeed-MII
pip install .[local]
pip install .
```

## Install DeepSpeed-GPT-NeoX
```bash
git clone -b ds-updates https://github.com/microsoft/deepspeed-gpt-neox.git
cd deepspeed-gpt-neox
pip install -r requirements/requirements-inference.txt
pip install .
python ./megatron/fused_kernels/setup.py install
cd ..
```

## Download Checkpoint
You can download the checkpoint file with:
```bash
wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/ -P 20B_checkpoints
```
or you can download with your favorite bittorrent client: [slim_weights.torrent](https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights.torrent)

Remember the location where you save the checkpoint directory and we will refer to this location as `{CKPT_DIR}`

---
📌 **Note:** The checkpoint file is nearly 40GB in size and may take a long time to download

---

## Run GPT-NeoX with MII
Modify the example file `examples/local/text-generation-neox-example.py`:
 - Change the `tensor_parallel` value in the `mii_config` dict to the number of GPUs on your system
 - Change the `local_model_path` in `mii.deploy()` call to `{CKPT_DIR}`

To run the example:
 - Start the server with `python3 examples/local/text-generation-neox-example.py`
 - Wait for the server to initialize
 - Run a query with `python3 examples/local/text-generation-query-example.py`
