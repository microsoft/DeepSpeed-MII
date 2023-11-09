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

    assert single_query_time == pytest.approx(double_query_time,
                                              0.1), "two queries should take about the same time as one query"


def test_query_kwargs(deployment, query):
    output = deployment(query, max_length=128, ignore_eos=True, top_p=0.9, top_k=50, temperature=0.9)
    assert output, "output is empty"
