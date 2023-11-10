# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest
import re
import subprocess
import time
import mii


def test_single_gpu(deployment, query):
    output = deployment(query)
    assert output, "output is empty"


def test_streaming(deployment, query):
    output = []

    def callback(response):
        output.append(response.response)

    deployment(query, streaming_fn=callback)
    assert output, "output is empty"


def test_multi_prompt(deployment, query):
    output = deployment([query] * 4)
    for r in output.response:
        assert r, "output is empty"


@pytest.mark.parametrize("tensor_parallel", [2])
def test_multi_gpu(deployment, query):
    output = deployment(query)
    assert output, "output is empty"


@pytest.mark.parametrize("replica_num", [2])
def test_multi_replica(deployment, query):
    deployment_name = deployment.mii_config.deployment_name

    start = time.time()
    output = mii.client(deployment_name)(query, max_length=128, ignore_eos=True)
    end = time.time()
    assert output, "output is empty"
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
        double_query_time, 0.1
    ), "two queries should take about the same time as one query"


def test_query_kwargs(deployment, query):
    # test ignore_eos
    output = deployment(query,
                        max_length=128,
                        min_new_tokens=16,
                        ignore_eos=True,
                        top_p=0.9,
                        top_k=50,
                        temperature=0.9)
    assert output, "output is empty"


def test_do_sample(deployment, query):
    output_0 = deployment(query, do_sample=False, max_length=128)
    output_1 = deployment(query, do_sample=False, max_length=128)
    assert output_0.response == output_1.response, "do_sample=False should always return the same output"


def test_stop_token(deployment, query):
    pytest.skip("not working yet")
    output = deployment(query, stop=".", max_length=512)
    print(str(output.response))
    assert str(output.response[0]).endswith("."), "output should end with 'the'"


def test_return_full_text(deployment, query):
    output = deployment(query, max_length=128, return_full_text=True)
    assert output.response[0].startswith(query), "output should start with the prompt"
