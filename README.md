[![Formatting](https://github.com/microsoft/DeepSpeed-MII/actions/workflows/formatting.yml/badge.svg?branch=main)](https://github.com/microsoft/DeepSpeed-MII/actions/workflows/formatting.yml)
[![nv-v100-legacy](https://github.com/microsoft/DeepSpeed-MII/actions/workflows/nv-v100-legacy.yml/badge.svg?branch=main)](https://github.com/microsoft/DeepSpeed-MII/actions/workflows/nv-v100-legacy.yml)
[![nv-a6000-fastgen](https://github.com/microsoft/DeepSpeed-MII/actions/workflows/nv-a6000-fastgen.yml/badge.svg?branch=main)](https://github.com/microsoft/DeepSpeed-MII/actions/workflows/nv-a6000-fastgen.yml)
[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/Microsoft/DeepSpeed/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/deepspeed-mii.svg)](https://pypi.org/project/deepspeed-mii/)
<!-- [![Documentation Status](https://readthedocs.org/projects/deepspeed/badge/?version=latest)](https://deepspeed.readthedocs.io/en/latest/?badge=latest) -->

<div align="center">
 <img src="docs/images/mii-white.svg#gh-light-mode-only" width="400px">
 <img src="docs/images/mii-dark.svg#gh-dark-mode-only" width="400px">
</div>

## Latest News

* [2024/01] [DeepSpeed-FastGen: Introducting Mixtral, Phi-2, and Falcon support with major performance and feature enhancements.](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen/2024-01-19)
* [2023/11] [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)
* [2022/11] [Stable Diffusion Image Generation under 1 second w. DeepSpeed MII](mii/legacy/examples/benchmark/txt2img)
* [2022/10] [Announcing DeepSpeed Model Implementations for Inference (MII)](https://www.deepspeed.ai/2022/10/10/mii.html)

# Contents

<!-- toc -->

- [DeepSpeed-MII](#deepspeed-mii)
- [Key Technologies](#key-technologies)
- [How does MII work?](#how-does-mii-work)
- [Supported Models](#supported-models)
- [Getting Started](#getting-started-with-mii)

<!-- tocstop -->

# DeepSpeed Model Implementations for Inference (MII) <a name="deepspeed-mii"></a>

Introducing MII, an open-source Python library designed by DeepSpeed to democratize powerful model inference with a focus on high-throughput, low latency, and cost-effectiveness.

* MII features include blocked KV-caching, continuous batching, Dynamic SplitFuse, tensor parallelism, and high-performance CUDA kernels to support fast high throughput text-generation for LLMs such as Llama-2-70B, Mixtral (MoE) 8x7B, and Phi-2. The latest updates in v0.2 add new model families, performance optimizations, and feature enhancements. MII now delivers up to 2.5 times higher effective throughput compared to leading systems such as vLLM. For detailed performance results please see our [latest DeepSpeed-FastGen blog](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen/2024-01-19) and [DeepSpeed-FastGen release blog](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen).

<div align="center">
 <img src="docs/images/fastgen-24-01-hero-light.png#gh-light-mode-only" width="850px">
 <img src="docs/images/fastgen-24-01-hero-dark.png#gh-dark-mode-only" width="850px">
</div>

<div align="center">
 <img src="docs/images/fastgen-hero-light.png#gh-light-mode-only" width="800px">
 <img src="docs/images/fastgen-hero-dark.png#gh-dark-mode-only" width="800px">
</div>

* We first [announced MII](https://www.deepspeed.ai/2022/10/10/mii.html) in 2022, which covers all prior releases up to v0.0.9. In addition to language models, we also support accelerating [text2image models like Stable Diffusion](examples/benchmark/txt2img). For more details on our previous releases please see our [legacy APIs](mii/legacy/).

# Key Technologies

## MII for High-Throughput Text Generation

MII provides accelerated text-generation inference through the use of four key technologies:

* Blocked KV Caching
* Continuous Batching
* Dynamic SplitFuse
* High Performance CUDA Kernels

For a deeper dive into understanding these features please [refer to our blog](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen) which also includes a detailed performance analysis.

## MII Legacy

In the past, MII introduced several [key performance optimizations](https://www.deepspeed.ai/2022/10/10/mii.html#inference-optimizations-with-mii) for low-latency serving scenarios:

* DeepFusion for Transformers
* Multi-GPU Inference with Tensor-Slicing
* ZeRO-Inference for Resource Constrained Systems
* Compiler Optimizations


# How does MII work?

<div align="center">
 <img src="docs/images/mii-arch-light.png#gh-light-mode-only" width="800px">
 <img src="docs/images/mii-arch-dark.png#gh-dark-mode-only" width="800px">
</div>


Figure 1: MII architecture, showing how MII automatically optimizes OSS models using DS-Inference before deploying them. DeepSpeed-FastGen optimizations in the figure have been published in [our blog post](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen).

Under-the-hood MII is powered by [DeepSpeed-Inference](https://github.com/microsoft/deepspeed). Based on the model architecture, model size, batch size, and available hardware resources, MII automatically applies the appropriate set of system optimizations to minimize latency and maximize throughput.


# Supported Models

MII currently supports over 20,000 models across eight popular model architectures. We plan to add additional models in the near term, if there are specific model architectures you would like supported please [file an issue](https://github.com/microsoft/DeepSpeed-MII/issues) and let us know. All current models leverage Hugging Face in our backend to provide both the model weights and the model's corresponding tokenizer. For our current release we support the following model architectures:

model family | size range | ~model count
------ | ------ | ------
[falcon](https://huggingface.co/models?other=falcon) | 7B - 180B | 300
[llama](https://huggingface.co/models?other=llama) | 7B - 65B | 19,000
[llama-2](https://huggingface.co/models?other=llama-2) | 7B - 70B | 900
[mistral](https://huggingface.co/models?other=mistral) | 7B | 6,000
[mixtral (MoE)](https://huggingface.co/models?other=mixtral) | 8x7B | 1,100
[opt](https://huggingface.co/models?other=opt) | 0.1B - 66B | 1,300
[phi-2](https://huggingface.co/models?other=phi) | 2.7B | 200
[qwen](https://huggingface.co/models?other=qwen) | 7B - 72B | 200

## MII Legacy Model Support

MII Legacy APIs support over 50,000 different models including BERT, RoBERTa, Stable Diffusion, and other text-generation models like Bloom, GPT-J, etc. For a full list please see our [legacy supported models table](mii/legacy/#supported-models-and-tasks).

# Getting Started with MII

DeepSpeed-MII allows users to create non-persistent and persistent deployments for supported models in just a few lines of code.

- [Installation](#installation)
- [Non-Persistent Pipeline](#non-persistent-pipeline)
- [Persistent Deployment](#persistent-deployment)

## Installation

The fasest way to get started is with our [PyPI release of DeepSpeed-MII](https://pypi.org/project/deepspeed-mii/) which means you can get started within minutes via:

```bash
pip install deepspeed-mii
```

For ease of use and significant reduction in lengthy compile times that many projects require in this space we distribute a pre-compiled python wheel covering the majority of our custom kernels through a new library called [DeepSpeed-Kernels](https://github.com/microsoft/DeepSpeed-Kernels). We have found this library to be very portable across environments with NVIDIA GPUs with compute capabilities 8.0+ (Ampere+), CUDA 11.6+, and Ubuntu 20+. In most cases you shouldn't even need to know this library exists as it is a dependency of DeepSpeed-MII and will be installed with it. However, if for whatever reason you need to compile our kernels manually please see our [advanced installation docs](https://github.com/microsoft/DeepSpeed-Kernels#source).

## Non-Persistent Pipeline

A non-persistent pipeline is a great way to try DeepSpeed-MII. Non-persistent pipelines are only around for the duration of the python script you are running. The full example for running a non-persistent pipeline deployment is only 4 lines. Give it a try!

```python
import mii
pipe = mii.pipeline("mistralai/Mistral-7B-v0.1")
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)
```

The returned `response` is a list of `Response` objects. We can access several details about the generation (e.g., `response[0].prompt_length`):

- `generated_text: str` Text generated by the model.
- `prompt_length: int` Number of tokens in the original prompt.
- `generated_length: int` Number of tokens generated.
- `finish_reason: str` Reason for stopping generation. `stop` indicates the EOS token was generated and `length` indicates the generation reached `max_new_tokens` or `max_length`.

If you want to free device memory and destroy the pipeline, use the `destroy` method:

```python
pipe.destroy()
```

### Tensor parallelism

Taking advantage of multi-GPU systems for greater performance is easy with MII. When run with the `deepspeed` launcher, tensor parallelism is automatically controlled by the `--num_gpus` flag:

```bash
# Run on a single GPU
deepspeed --num_gpus 1 mii-example.py

# Run on multiple GPUs
deepspeed --num_gpus 2 mii-example.py
```
### Pipeline Options
While only the model name or path is required to stand up a non-persistent pipeline deployment, we offer customization options to our users:

**`mii.pipeline()` Options**:
- `model_name_or_path: str` Name or local path to a [HuggingFace](https://huggingface.co/) model.
- `max_length: int` Sets the default maximum token length for the prompt + response.
- `all_rank_output: bool` When enabled, all ranks return the generated text. By default, only rank 0 will return text.

Users can also control the generation characteristics for individual prompts (i.e., when calling `pipe()`) with the following options:

- `max_length: int` Sets the per-prompt maximum token length for prompt + response.
- `min_new_tokens: int` Sets the minimum number of tokens generated in the response. `max_length` will take precedence over this setting.
- `max_new_tokens: int` Sets the maximum number of tokens generated in the response.
- `ignore_eos: bool` (Defaults to `False`) Setting to `True` prevents generation from ending when the EOS token is encountered.
- `top_p: float` (Defaults to `0.9`) When set below `1.0`, filter tokens and keep only the most probable, where token probabilities sum to &ge;`top_p`.
- `top_k: int` (Defaults to `None`) When `None`, top-k filtering is disabled. When set, the number of highest probability tokens to keep.
- `temperature: float` (Defaults to `None`) When `None`, temperature is disabled. When set, modulates token probabilities.
- `do_sample: bool` (Defaults to `True`) When `True`, sample output logits. When `False`, use greedy sampling.
- `return_full_text: bool` (Defaults to `False`) When `True`, prepends the input prompt to the returned text

## Persistent Deployment

A persistent deployment is ideal for use with long-running and production applications. The persistent model uses a lightweight GRPC server that can be queried by multiple clients at once. The full example for running a persistent model is only 5 lines. Give it a try!

```python
import mii
client = mii.serve("mistralai/Mistral-7B-v0.1")
response = client.generate(["Deepspeed is", "Seattle is"], max_new_tokens=128)
print(response)
```

The returned `response` is a list of `Response` objects. We can access several details about the generation (e.g., `response[0].prompt_length`):

- `generated_text: str` Text generated by the model.
- `prompt_length: int` Number of tokens in the original prompt.
- `generated_length: int` Number of tokens generated.
- `finish_reason: str` Reason for stopping generation. `stop` indicates the EOS token was generated and `length` indicates the generation reached `max_new_tokens` or `max_length`.

If we want to generate text from other processes, we can do that too:

```python
client = mii.client("mistralai/Mistral-7B-v0.1")
response = client.generate("Deepspeed is", max_new_tokens=128)
```

When we no longer need a persistent deployment, we can shutdown the server from any client:

```python
client.terminate_server()
```

### Model Parallelism

Taking advantage of multi-GPU systems for better latency and throughput is also easy with the persistent deployments. Model parallelism is controlled by the `tensor_parallel` input to `mii.serve`:

```python
client = mii.serve("mistralai/Mistral-7B-v0.1", tensor_parallel=2)
```

The resulting deployment will split the model across 2 GPUs to deliver faster inference and higher throughput than a single GPU.

### Model Replicas

We can also take advantage of multi-GPU (and multi-node) systems by setting up multiple model replicas and taking advantage of the load-balancing that DeepSpeed-MII provides:

```python
client = mii.serve("mistralai/Mistral-7B-v0.1", replica_num=2)
```

The resulting deployment will load 2 model replicas (one per GPU) and load-balance incoming requests between the 2 model instances.

Model parallelism and replicas can also be combined to take advantage of systems with many more GPUs. In the example below, we run 2 model replicas, each split across 2 GPUs on a system with 4 GPUs:

```python
client = mii.serve("mistralai/Mistral-7B-v0.1", tensor_parallel=2, replica_num=2)
```

The choice between model parallelism and model replicas for maximum performance will depend on the nature of the hardware, model, and workload. For example, with small models users may find that model replicas provide the lowest average latency for requests. Meanwhile, large models may achieve greater overall throughput when using only model parallelism.

### RESTful API

MII makes it easy to setup and run model inference via RESTful APIs by setting `enable_restful_api=True` when creating a persistent MII deployment. The RESTful API can receive requests at `http://{HOST}:{RESTFUL_API_PORT}/mii/{DEPLOYMENT_NAME}`. A full example is provided below:

```python
client = mii.serve(
    "mistralai/Mistral-7B-v0.1",
    deployment_name="mistral-deployment",
    enable_restful_api=True,
    restful_api_port=28080,
)
```

---
📌 **Note:** While providing a `deployment_name` is not necessary (MII will autogenerate one for you), it is good practice to provide a `deployment_name` so that you can ensure you are interfacing with the correct RESTful API.

---

You can then send prompts to the RESTful gateway with any HTTP client, such as `curl`:

```bash
curl --header "Content-Type: application/json" --request POST  -d '{"prompts": ["DeepSpeed is", "Seattle is"], "max_length": 128}' http://localhost:28080/mii/mistral-deployment
```

or `python`:

```python
import json
import requests
url = f"http://localhost:28080/mii/mistral-deployment"
params = {"prompts": ["DeepSpeed is", "Seattle is"], "max_length": 128}
json_params = json.dumps(params)
output = requests.post(
    url, data=json_params, headers={"Content-Type": "application/json"}
)
```

<!--
### Token Streaming
With a persistent deployment, the resulting response text can be streamed back to the client as it is generated. This functionality is useful for chatbot style applications. A simple example of streaming tokens is below:
```python
import mii

out_tokens = []
def callback(response):
    print(f"Received: {response.response}")
    out_tokens.append(response.response)

client = mii.serve("mistralai/Mistral-7B-v0.1")
client.generate("Deepspeed is", streaming_fn=callback)
```

To enable streaming output, we must provide `streaming_fn` with the prompt. This should be a callable function that acts as a callback and will receive the streaming tokens at they are generated. In the example above, we show a simple function that prints the current token and appends to a final output `out_tokens`.
-->

### Persistent Deployment Options
While only the model name or path is required to stand up a persistent deployment, we offer customization options to our users.

**`mii.serve()` Options**:
- `model_name_or_path: str` (Required) Name or local path to a [HuggingFace](https://huggingface.co/) model.
- `max_length: int` (Defaults to maximum sequence length in model config) Sets the default maximum token length for the prompt + response.
- `deployment_name: str` (Defaults to `f"{model_name_or_path}-mii-deployment"`) A unique identifying string for the persistent model. If provided, client objects should be retrieved with `client = mii.client(deployment_name)`.
- `tensor_parallel: int` (Defaults to `1`) Number of GPUs to split the model across.
- `replica_num: int` (Defaults to `1`) The number of model replicas to stand up.
- `enable_restful_api: bool` (Defaults to `False`) When enabled, a RESTful API gateway process is launched that can be queried at `http://{host}:{restful_api_port}/mii/{deployment_name}`. See the [section on RESTful APIs](#restful-api) for more details.
- `restful_api_port: int` (Defaults to `28080`) The port number used to interface with the RESTful API when `enable_restful_api` is set to `True`.

**`mii.client()` Options**:
- `model_or_deployment_name: str` Name of the model or `deployment_name` passed to `mii.serve()`

Users can also control the generation characteristics for individual prompts (i.e., when calling `client.generate()`) with the following options:

- `max_length: int` Sets the per-prompt maximum token length for prompt + response.
- `min_new_tokens: int` Sets the minimum number of tokens generated in the response. `max_length` will take precedence over this setting.
- `max_new_tokens: int` Sets the maximum number of tokens generated in the response.
- `ignore_eos: bool` (Defaults to `False`) Setting to `True` prevents generation from ending when the EOS token is encountered.
- `top_p: float` (Defaults to `0.9`) When set below `1.0`, filter tokens and keep only the most probable, where token probabilities sum to &ge;`top_p`.
- `top_k: int` (Defaults to `None`) When `None`, top-k filtering is disabled. When set, the number of highest probability tokens to keep.
- `temperature: float` (Defaults to `None`) When `None`, temperature is disabled. When set, modulates token probabilities.
- `do_sample: bool` (Defaults to `True`) When `True`, sample output logits. When `False`, use greedy sampling.
- `return_full_text: bool` (Defaults to `False`) When `True`, prepends the input prompt to the returned text


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
