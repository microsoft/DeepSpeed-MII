# Generate Stable Diffusion Images with < 1 Second per Image

<div align="center">
 <img src="../../docs/images/sd-hero.png#gh-light-mode-only">
 <img src="../../docs/images/sd-hero-dark.png#gh-dark-mode-only">
</div>

In this tutorial you will learn how to deploy [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) with state-of-the-art performance optimizations from DeepSpeed. Specifically, you will use both [DeepSpeed Inference](https://github.com/microsoft/deepspeed) and [DeepSpeed-MII](https://github.com/microsoft/deepspeed-mii) to your deployment. In addition to deploying we will perform several performance evaluations.

This tutorial and related results are using Azure [ND96amsr\_A100\_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series) instances with NVIDIA A100-80GB GPUs. We observe similar performance on [ND96asr\_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series) instances with NVIDIA A100-40GB GPUs. In addition all of the techniques described here have also been successfully deployed on NVIDIA RTX A6000 GPUs as well.

## Outline
* [Environment and dependency setup](#environment-setup)
* [Deploy and evaluate baseline Stable Diffusion with diffusers](#deploy-baseline-stable-diffusion-with-diffusers)
* [Deploy and evaluate MII with DeepSpeed inference optimizations](#deploy-mii-with-deepspeed-inference-optimizations)

## Environment and dependency setup

Install [DeepSpeed](https://pypi.org/project/deepspeed/) and [DeepSpeed-MII](https://pypi.org/project/mii/) via pip. For this tutorial you'll want to include the "sd" extra with DeepSpeed, this will add a few extra dependencies to enable the optimizations in this tutorial.

```bash
pip install deepspeed[sd] deepspeed-mii
```

In order to check your DeepSpeed install is setup correctly run `ds_report` from your command line. This will show what versions of DeepSpeed, PyTorch, and nvcc will be used at runtime. The bottom half of `ds_report` is show below for our setup:

```
DeepSpeed general environment info:
torch install path ............... ['/usr/local/lib/python3.9/dist-packages/torch']
torch version .................... 1.12.1+cu116
torch cuda version ............... 11.6
torch hip version ................ None
nvcc version ..................... 11.6
deepspeed install path ........... ['/usr/local/lib/python3.9/dist-packages/deepspeed']
deepspeed info ................... 0.7.4, unknown, unknown
deepspeed wheel compiled w. ...... torch 1.12, cuda 11.6
```

You can see we are running PyTorch 1.12.1 built against CUDA 11.6 and our NVCC version of 11.6 is properly aligned with the installed torch version.

Some additional environment context for reproducibility:
* Ubuntu 20.04.4 LTS
* Python 3.9.15
* deepspeed==0.7.4
* deepspeed-mii==0.0.3
* torch==1.12.1+cu116
* diffusers==0.6.0
* transformers==4.23.1
* triton==2.0.0.dev20221005

## Deploy baseline Stable Diffusion with diffusers

Let's first deploy the baseline Stable Diffusion from the [diffusers tutorial](https://github.com/huggingface/diffusers#text-to-image-generation-with-stable-diffusion). We've modified their example to use an explicit auth token for downloading the model, you can get your auth token from your account on the [Hugging Face Hub](https://huggingface.co/settings/tokens). If you do not already have one, you can create a token by going to your [Hugging Face Settings](https://huggingface.co/settings/tokens) and clicking on the `New Token` button. You will also need to accept the license of [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) to be able to download it.

Going forward we will refer to [baseline-sd.py]() to run and benchmark a non-MII accelerated baseline.

We utilize the `StableDiffusionPipeline` from diffusers to download and setup the model and move it to our GPU via:

```python
hf_auth_key = "hf_xxxxxxxxxxx"
pipe = diffusers.StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=hf_auth_key,
    torch_dtype=torch.float16,
    revision="fp16").to("cuda")
```

In general we're able to use this `pipe` to generate an image from text prompts, here is an example:

```python
image = pipe("a photo of an astronaut riding a horse on mars").images[0]
image.save("horse-on-mars.png")
```

For your convience we've created a runnable script that sets up the pipeline, runs an example, and runs a benchmarks. You can run this example via:

```bash
export HF_AUTH_TOKEN=hf_xxxxxxxx
python baseline-sd.py
```

We've created a helper benchmark utility in [utils.py]() that adds basic timing around each image generation, prints the results, and saves the images.

You can modify the `baseline-sd.py` script to use different batch sizes, in this case we will run batch size 1 to evaluate a latency sensitive scenario.

Here is what we observe in terms of performance over 5 trials with the same prompt:

```
100%|███████████████████████████████████████████████| 51/51 [00:02<00:00, 23.24it/s]
trial=0, time_taken=2.3496
100%|███████████████████████████████████████████████| 51/51 [00:02<00:00, 23.46it/s]
trial=1, time_taken=2.3371
100%|███████████████████████████████████████████████| 51/51 [00:02<00:00, 23.52it/s]
trial=2, time_taken=2.3185
100%|███████████████████████████████████████████████| 51/51 [00:02<00:00, 23.54it/s]
trial=3, time_taken=2.3274
100%|███████████████████████████████████████████████| 51/51 [00:02<00:00, 23.57it/s]
trial=4, time_taken=2.3148
```

## Deploy MII with DeepSpeed inference optimizations

Next we will use DeepSpeed-MII to setup a persistent deployment that utilizes DeepSpeed inference optimizations to improve latency by 1.8x.

As of today, DeepSpeed inference will automatically apply several different optimizations to Stable Diffusion:

1. FlashAttention for UNet cross-attention
    * We created an adapted version of [Triton](https://github.com/openai/triton)'s FlashAttention to support our inference-only scenario plus Stable Diffusion specific requirements.
    * We compared both the Triton and original implementation and found our Triton version performed better for small batch sizes.
2. Custom CUDA kernels for UNet cross-attention
3. [nvFuser](https://pytorch.org/blog/introducing-nvfuser-a-deep-learning-compiler-for-pytorch/)
4. [CUDA Graph](https://developer.nvidia.com/blog/cuda-graphs/)
5. UNet channel-last memory format

Keep an eye on the [DeepSpeed-MII](https://github.com/microsoft/deepspeed-mii) repo and this tutorial for further updates and a deeper dive into these and future performance optimizations to achieve *under 1 second* generation times in latency sensitive deployments and further improvements to multi-batch throughput.

Stable Diffusion can be deployed with MII in the following way. You provide your Hugging Face auth key in an `mii_config` and tell MII what model and task you want to deploy in the `mii.deploy` API.

```python
import mii

mii_config = {
    "dtype": "fp16",
    "hf_auth_token": "hf_xxxxxxxxxxxxxxx"
}

mii.deploy(task='text-to-image',
           model="CompVis/stable-diffusion-v1-4",
           deployment_name="sd_deploy",
           mii_config=mii_config)
```

The above code will deploy Stable Diffusion on your local machine using the DeepSpeed inference open-source optimizations listed above. It will keep the deployment **persistent** and expose a gRPC interface for you to make repeated queries via command-line or from custom applications. See below for how to make queries to your MII deployment:

```python
import mii
generator = mii.mii_query_handle("sd_deploy")
prompt = "a photo of an astronaut riding a horse on mars"
image = generator.query({'query': prompt}).images[0]
image.save("horse-on-mars.png")
```

We've packaged up all that you need to deploy, query, and tear down an SD MII deployment for you in [mii-sd.py]() which we will refer to going forward. You can run this example via:

```bash
export HF_AUTH_TOKEN=hf_xxxxxxxx
python mii-sd.py
```

We use the same helper benchmark utility in [utils.py]() as we did in the baseline to evaluate the MII deployment.

Similar to baseline you can modify the `mii-sd.py` script to use different batch sizes, for comparison purposes we run with batch size 1 to evaluate a latency sensitive scenario.

Here is what we observe in terms of performance over 5 trials with the same prompt:

```
100%|███████████████████████████████████████████████| 51/51 [00:01<00:00, 43.58it/s]
trial=0, time_taken=1.2683
100%|███████████████████████████████████████████████| 51/51 [00:01<00:00, 43.69it/s]
trial=1, time_taken=1.2635
100%|███████████████████████████████████████████████| 51/51 [00:01<00:00, 43.69it/s]
trial=2, time_taken=1.2683
100%|███████████████████████████████████████████████| 51/51 [00:01<00:00, 43.67it/s]
trial=3, time_taken=1.2786
100%|███████████████████████████████████████████████| 51/51 [00:01<00:00, 43.72it/s]
trial=4, time_taken=1.2626
```
