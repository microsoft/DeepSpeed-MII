# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest

import json
import re
import requests
import subprocess
import time

import mii


def test_single_gpu(deployment, query):
    outputs = deployment(query)
    assert outputs[0], "output is empty"


def test_streaming(deployment, query):
    outputs = []

    def callback(response):
        outputs.append(response[0].generated_text)

    deployment(query, streaming_fn=callback)
    assert outputs, "output is empty"


def test_streaming_consistency(deployment, query):
    expected_output = deployment(query, do_sample=False)
    streaming_parts = []

    def callback(response):
        streaming_parts.append(response[0].generated_text)

    deployment(query, do_sample=False, streaming_fn=callback)
    streaming_output = "".join(streaming_parts)

    assert streaming_output == expected_output[0].generated_text, "outputs w and w/o streaming are not equal"


def test_multi_prompt(deployment, query):
    outputs = deployment([query] * 4)
    for r in outputs:
        assert r, "output is empty"


@pytest.mark.parametrize("tensor_parallel", [2])
def test_multi_gpu(deployment, query):
    outputs = deployment(query)
    assert outputs[0], "output is empty"


@pytest.mark.parametrize("replica_num", [2])
def test_multi_replica(deployment, query):
    deployment_name = deployment.mii_config.deployment_name

    start = time.time()
    outputs = mii.client(deployment_name)(query, max_length=128, ignore_eos=True)
    end = time.time()
    assert outputs[0], "output is empty"
    single_query_time = end - start

    procs = []
    double_query_time = []
    for _ in range(2):
        p = subprocess.Popen(
            [
                "python3",
                "-c",
                f"import time, mii; start=time.time(); mii.client('{deployment_name}')('{query}', max_length=128, ignore_eos=True); print('time',time.time()-start)",
            ],
            stdout=subprocess.PIPE,
        )
        procs.append(p)
    for p in procs:
        output, error = p.communicate()
        m = re.search(r"time (\d+.\d+)", output.decode("utf-8").strip())
        assert m, "time not found"
        double_query_time.append(float(m.group(1)))

    double_query_time = sum(double_query_time) / 2

    assert single_query_time == pytest.approx(
        double_query_time, single_query_time / 2
    ), "two queries should take about the same time as one query"


def test_query_kwargs(deployment, query):
    # test ignore_eos
    outputs = deployment(
        query,
        max_length=128,
        min_new_tokens=16,
        ignore_eos=True,
        top_p=0.9,
        top_k=50,
        temperature=0.9,
    )
    assert outputs[0], "output is empty"


def test_do_sample(deployment, query):
    output_0 = deployment(query, do_sample=False, max_length=128)
    output_1 = deployment(query, do_sample=False, max_length=128)
    assert (
        output_0[0] == output_1[0]
    ), "do_sample=False should always return the same output"


def test_return_full_text(deployment, query):
    outputs = deployment(query, max_length=128, return_full_text=True)
    assert outputs[0].generated_text.startswith(query), "output should start with the prompt"


@pytest.mark.parametrize("enable_restful_api", [True])
def test_restful_api(deployment, query, deployment_name, restful_api_port):
    # Verify deployment is running
    outputs = deployment(query, max_length=128)
    assert outputs[0], "output is empty"

    # Verify REST API
    url = f"http://localhost:{restful_api_port}/mii/{deployment_name}"
    params = {"prompts": query, "max_length": 128}
    json_params = json.dumps(params)
    result = requests.post(url,
                           data=json_params,
                           headers={"Content-Type": "application/json"})
    assert result.status_code == 200
    assert "generated_text" in result.json()[0]
