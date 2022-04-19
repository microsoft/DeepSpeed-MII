<div align="center">
 <img src="docs/images/mii-white.svg#gh-light-mode-only" width="400px">
 <img src="docs/images/mii-dark.svg#gh-dark-mode-only" width="400px">
</div>

Model Implementations for Inference (MII) is a new open-source python library from DeepSpeed, designed to make low-latency, low-cost inference of powerful transformer models not only feasible but also easily accessible. It does so by offering access to highly optimized implementation of thousands of widely used DL models. In fact, straight out-of-box, MII supported models can deployed on-premise or on Azure through AML with just a few lines of code while achieving a latency and cost reduction of up to Nx

TODO: add figure showing latency reductions across many models w. MII over baseline

## How does MII work?

Under-the-hood MII is powered by DeepSpeed-Inference. Based on model type, model size, batch size, and available hardware resources, MII automatically applies the appropriate set of system optimizations from DeepSpeed-Inference to minimize latency and maximize thoughput. It does so using one of many pre-specified model injection policies, that allows DeepSpeed-Inference to identify the underlying pytorch model architecture and replace it with an optimized implementation. This injection can replace a single GPU module with multi-GPU variations enabling models to run on single GPU device, or seamlessly scale to tens of GPUs for dense models and hundreds of GPUs for sparse models for lower latency and higher throughput.

MII makes the expansive set of optimizations in DeepSpeed-Inference easily accessible to its users by automatically integrating them to thousands of popular transformer models. For a full set of optimizations in DeepSpeed-Inference please see here (TODO: cite ds-inference paper or blog)

## Supported Models and Tasks

MII supports a growing list of tasks such as text-generation, question-answering, text-classification, etc, across thousands of transformer models available through multiple open-sourced model repositories such as Hugging Face, FairSeq, EluetherAI, etc. It supports dense models based on Bert, Roberta or GPT architectures ranging from few hundred million parameters to tens of billions of parameters in size. We continue to expand the list with support for massive hundred billion plus parameter dense and sparse models coming soon.

For a full set of models and tasks supported by MII, please see here (TODO: add reference to specific model classes we support)

## Getting Started with MII

### Installation

#### Local deployments
`pip install .[local]` will install all dependencies required for a local deployment.

#### Azure deployments
`pip install .[azure]` will install all python-based dependencies required for a local deployment.

Azure local deployments require:
* nvidia-docker

### Deploying with MII

MII allows supported models to be deployed with just a few lines of code both on-premise and on Azure.

#### On-Premises Deployment

TODO: add example code for local deployment

#### Azure Deployment

Azure-Local Deployment

Azure-on-AKS-Deployment


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
