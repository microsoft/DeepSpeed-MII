# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest

from mii.batching.ragged_batching import ReadableStream
from mii.config import ModelConfig
from mii.modeling.tokenizers import load_tokenizer


@pytest.mark.parametrize(
    "model_name",
    [
        "tiiuae/falcon-7b",
        "NousResearch/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
        "cloudyu/Mixtral_11Bx2_MoE_19B",
        "facebook/opt-125m",
    ],
    ids=["falcon",
         "llama",
         "mistral",
         "mixtral",
         "opt"],
)
@pytest.mark.parametrize(
    "query",
    [
        "It’s a region that includes Washington, Oregon, and Idaho.",
        "# Heading\n\n<s>title</s>   redundant  spaces, #id — an anchor",
        "例如",
    ],
    ids=[
        "apostrophe",
        "markdown",
        "chinese",
    ])
def test_readable_stream(model_config, query):
    tokenizer = load_tokenizer(ModelConfig(**model_config))
    thread_id = 42

    token_ids = tokenizer.encode(query)
    expected = tokenizer.decode(token_ids)
    decoded = []

    stream = ReadableStream(tokenizer)
    for token_id in token_ids:
        decoded.append(stream.decode(thread_id, [token_id]))

    assert "".join(decoded) == expected
