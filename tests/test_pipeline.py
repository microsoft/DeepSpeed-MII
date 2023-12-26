# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


def test_single_gpu(pipeline, query):
    outputs = pipeline(query)
    assert outputs[0], "output is empty"


def test_multi_prompt(pipeline, query):
    outputs = pipeline([query] * 4)
    for r in outputs:
        assert r, "output is empty"


def test_query_kwargs(pipeline, query):
    # test ignore_eos
    outputs = pipeline(
        query,
        max_length=128,
        min_new_tokens=16,
        ignore_eos=True,
        top_p=0.9,
        top_k=50,
        temperature=0.9,
    )
    assert outputs[0], "output is empty"
