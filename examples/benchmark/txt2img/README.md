# Stable Diffusion Image Generation under 1 second w. DeepSpeed MII

<div align="center">
 <img src="../../../docs/images/sd-hero-light.png#gh-light-mode-only">
 <img src="../../../docs/images/sd-hero-dark.png#gh-dark-mode-only">
</div>

In this tutorial you will learn how to deploy [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) with state-of-the-art performance optimizations from [DeepSpeed Inference](https://github.com/microsoft/deepspeed) and [DeepSpeed-MII](https://github.com/microsoft/deepspeed-mii). In addition to deploying we will perform several performance evaluations.

The performance results above utilized NVIDIA GPUs from Azure: [ND96amsr\_A100\_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series) (NVIDIA A100-80GB) and [ND96asr\_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series) (A100-40GB). We have also used MII-Public with NVIDIA RTX-A6000 GPUs and will include those results at a future date.

## Outline
* [Optimizations for Stable Diffusion with DeepSpeed-MII](#optimizations)
* [Environment and dependency setup](#environment-and-dependency-setup)
* [Evaluation methodology](#evaluation-methodology)
* [Deploy baseline Stable Diffusion with diffusers](#deploy-baseline-stable-diffusion-with-diffusers)
* [Deploy Stable Diffusion with MII-Public](#deploy-mii-with-MII-Public)
* [Deploy Stable Diffusion with MII-Azure](#deploy-mii-with-MII-Azure)

## Stable Diffusion Optimizations with DeepSpeed-MII

DeepSpeed-MII will automatically inject a wide range of optimizations from DeepSpeed-Inference to accelerate Stable Diffusion Deployment. We list the optimizations below:

1. FlashAttention for UNet cross-attention
    * The implementation is adapted from [Triton](https://github.com/openai/triton)'s FlashAttention and further tuned to accelerate Stable Diffusion specific scenarios.
2. UNet channel-last memory format
    * Faster convolution performance using NHWC data layout
    * Removal of NHWC <--> NCHW data layout conversion through NHWC implementation of missing operators
3. Custom CUDA implementations of:
    * LayerNorm
    * Cross-attention
4. [CUDA Graph](https://developer.nvidia.com/blog/cuda-graphs/) for VAE, UNet, and CLIP encoder
5. Custom CUDA implementation of:
   * GroupNorm
   * Fusion across multiple elementwise operators
6. Partial UNet INT8 quantization via [ZeroQuant](https://arxiv.org/abs/2206.01861)
7. Exploitation of coarse grained computation sparsity

The first four optimizations are available via MII-Public, while the rest are available via MII-Azure ([see here to read more about MII-Public and MII-Azure](https://github.com/microsoft/deepspeed-mii#mii-public-and-mii-azure)). In the rest of this tutorial, we will show how you can deploy Stable Diffusion with both MII-Public and MII-Azure.

Keep an eye on the [DeepSpeed-MII](https://github.com/microsoft/deepspeed-mii) repo and this tutorial for further updates and a deeper dive into these and future performance optimizations.

## Environment and dependency setup

Install [DeepSpeed](https://pypi.org/project/deepspeed/) and [DeepSpeed-MII](https://pypi.org/project/mii/) via pip. For this tutorial you'll want to include the "sd" extra with DeepSpeed, this will add a few extra dependencies to enable the optimizations in this tutorial.

```bash
pip install deepspeed[sd] deepspeed-mii
```

> **Note**
> The DeepSpeed version used in the rest of this tutorial uses [this branch](https://github.com/microsoft/DeepSpeed/pull/2491) which will be merged into master and released as part of DeepSpeed v0.7.5 later this week.

In order to check your DeepSpeed install is setup correctly run `ds_report` from your command line. This will show what versions of DeepSpeed, PyTorch, and nvcc will be used at runtime. The bottom half of `ds_report` is show below for our setup:

```
DeepSpeed general environment info:
torch install path ............... ['/usr/local/lib/python3.9/dist-packages/torch']
torch version .................... 1.12.1+cu116
torch cuda version ............... 11.6
torch hip version ................ None
nvcc version ..................... 11.6
deepspeed install path ........... ['/usr/local/lib/python3.9/dist-packages/deepspeed']
deepspeed info ................... 0.7.5, unknown, unknown
deepspeed wheel compiled w. ...... torch 1.12, cuda 11.6
```

You can see we are running PyTorch 1.12.1 built against CUDA 11.6 and our NVCC version of 11.6 is properly aligned with the installed torch version, this alignment is highly recommended by both us and NVIDIA.

The [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) which includes `nvcc` is required for DeepSpeed. All of our custom CUDA/C++ ops for this tutorial will be compiled the first time you run the model in under 30 seconds. By comparison some similar projects can take upwards of 30+ minutes to compile and get running.

Some additional environment context for reproducibility:
* deepspeed==0.7.5
* deepspeed-mii==0.0.3
* torch==1.12.1+cu116
* diffusers==0.7.1
* transformers==4.24.0
* triton==2.0.0.dev20221005
* Ubuntu 20.04.4 LTS
* Python 3.9.15

## Evaluation methodology

The evaluations in this tutorial are done with the full `diffusers` end-to-end pipeline. The primary steps of the pipeline include CLIP, UNet, VAE, safety checker, and PIL conversion. We see that in `diffusers` v0.7.1 the resulting image tensors are moved from GPU to CPU for both safety checking and PIL conversion ([essentially this code block](https://github.com/huggingface/diffusers/blob/v0.7.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L406-L420)). We observe between 40-60 ms for the safety checker and PIL conversion in our setup but see these times can vary more dramatically in shared CPU environments.

We run the same text prompt for single and multi-batch cases. In addition, we run each method for 10 trials and report the median value.

## Deploy baseline Stable Diffusion with diffusers

Let's first deploy the baseline Stable Diffusion from the [diffusers tutorial](https://github.com/huggingface/diffusers#text-to-image-generation-with-stable-diffusion). In this example we will show performance on a NVIDIA A100-40GB GPU from Azure. We've modified their example to use an explicit auth token for downloading the model, you can get your auth token from your account on the [Hugging Face Hub](https://huggingface.co/settings/tokens). If you do not already have one, you can create a token by going to your [Hugging Face Settings](https://huggingface.co/settings/tokens) and clicking on the `New Token` button. You will also need to accept the license of [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) to be able to download it.

Going forward we will refer to [baseline-sd.py](baseline-sd.py) to run and benchmark a non-MII accelerated baseline.

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

We use the `diffusers` [StableDiffusionPipeline](https://huggingface.co/docs/diffusers/v0.7.0/en/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline.__call__) defaults for image height/width (512x512) and number of inference steps (50). Changing these values will impact the the latency/throughput you will see.

For your convenience we've created a runnable script that sets up the pipeline, runs an example, and runs a benchmarks. You can run this example via:

```bash
export HF_AUTH_TOKEN=hf_xxxxxxxx
python baseline-sd.py
```

We've created a helper benchmark utility in [utils.py](utils.py) that adds basic timing around each image generation, prints the results, and saves the images.

You can modify the `baseline-sd.py` script to use different batch sizes, in this case we will run batch size 1 to evaluate a latency sensitive scenario.

Here is what we observe in terms of A100-40GB performance over 10 trials with the same prompt:

```
trial=0, time_taken=2.2166
trial=1, time_taken=2.2143
trial=2, time_taken=2.2205
trial=3, time_taken=2.2106
trial=4, time_taken=2.2105
trial=5, time_taken=2.2254
trial=6, time_taken=2.2044
trial=7, time_taken=2.2304
trial=8, time_taken=2.2078
trial=9, time_taken=2.2097
median duration: 2.2125
```

## Deploy Stable diffusion with MII-Public

MII-Public improves latency by up to 1.8x compared to several baselines (see image at top of tutorial). To create a MII-Public deployment, simply provide your Hugging Face auth key in an `mii_config` and tell MII what model and task you want to deploy in the `mii.deploy` API.

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

We've packaged up all that you need to deploy, query, and tear down an SD MII deployment in [mii-sd.py](mii-sd.py) which we will refer to going forward. You can run this example via:

```bash
export HF_AUTH_TOKEN=hf_xxxxxxxx
python mii-sd.py
```

We use the same helper benchmark utility in [utils.py](utils.py) as we did in the baseline to evaluate the MII deployment.

Similar to baseline you can modify the `mii-sd.py` script to use different batch sizes, for comparison purposes we run with batch size 1 to evaluate a latency sensitive scenario.

Here is what we observe in terms of A100-40GB performance over 10 trials with the same prompt:

```
trial=0, time_taken=1.3935
trial=1, time_taken=1.2091
trial=2, time_taken=1.2110
trial=3, time_taken=1.2068
trial=4, time_taken=1.2064
trial=5, time_taken=1.2002
trial=6, time_taken=1.2063
trial=7, time_taken=1.2062
trial=8, time_taken=1.2069
trial=9, time_taken=1.2063
median duration: 1.2065
```
## Deploy Stable Diffusion with MII-Azure

Continue to watch this space for updates in the coming weeks on how to get access to MII-Azure. We will be providing two options for users to access these optimizations: (1) Azure VM Image and (2) AzureML endpoint deployments.

An Azure VM image will be released via the [Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/) in late November 2022. This VM will have MII-Azure pre-installed with all the required components to get started with models like Stable Diffusion and other MII supported models. The VM image itself will be free and can be used with any Azure Compute Instances.

A more comprehensive [AzureML (AML)](https://azure.microsoft.com/en-us/free/machine-learning/) deployment with MII-Azure is also in the works to make deploying on AML with MII-Azure quick and easy to use. Keep watching our MII repo for more updates on this release.
