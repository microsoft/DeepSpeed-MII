# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
deployment = \
"""$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: <deployment-name>
endpoint_name: <endpoint-name>
model:
    path: <model-path>
model_mount_path: /var/azureml-model
code_configuration:
  code: <code-path>
  scoring_script: score.py
environment: azureml:<environment-name>:<version>
environment_variables:
  AML_APP_ROOT: /var/azureml-model/code
  WORKER_TIMEOUT: 2400
  WORKER_COUNT: <replica-num>
  AZUREML_LOG_LEVEL: DEBUG
  LOG_IO: 1
instance_type: <instance-type>
request_settings:
  request_timeout_ms: 90000
  max_concurrent_requests_per_instance: <replica-num>
liveness_probe:
  initial_delay: 300
  timeout: 1
  period: 60
  success_threshold: 1
  failure_threshold: 40
readiness_probe:
  initial_delay: 300
  timeout: 1
  period: 60
  success_threshold: 1
  failure_threshold: 40
instance_count: 1
"""

endpoint = \
"""$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: <endpoint-name>
auth_mode: key
"""

environment = \
"""$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: <environment-name>
version: <version>
image: <acr-name>.azurecr.io/<image-name>:<version>
inference_config:
  liveness_route:
    path: /
    port: 5001
  readiness_route:
    path: /
    port: 5001
  scoring_route:
    path: /score
    port: 5001
"""

model_download = \
"""import os
import glob
import shutil

# Path and model params
model_path = "<model-path>"
tmp_download_path = "./tmp/"
snapshot_rel_path = "*/snapshots/*/*"
model = "<model-name>"
task = "<task-name>"

# Must set cache location before loading transformers
os.environ["TRANSFORMERS_CACHE"] = tmp_download_path

from transformers import pipeline
from huggingface_hub import snapshot_download

# Download model
try:
    _ = pipeline(task=task, model=model)
except OSError:
    # Sometimes the model cannot be downloaded and we need to grab the snapshot
    snapshot_download(model, cache_dir=tmp_download_path)

# We need to resolve symlinks and move files to model_path dir
os.mkdir(model_path)
for f_path in glob.glob(os.path.join(tmp_download_path, snapshot_rel_path)):
    f_name = os.path.basename(f_path)
    real_file = os.path.realpath(f_path)
    new_file = os.path.join(model_path, f_name)
    os.rename(real_file, new_file)

shutil.rmtree(tmp_download_path)
"""

deploy = \
"""set -e
python3 model_download.py
az acr build -r <acr-name> --build-arg no-cache=True -t "<image-name>:<version>" build
az ml environment create -f environment.yml
az ml online-endpoint create -n "<endpoint-name>" -f endpoint.yml
az ml online-deployment create -n "<deployment-name>" -f deployment.yml
"""

dockerfile = \
        """FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV AML_APP_ROOT=/var/azureml-model/code \
    BUILD_DIR=/tmp/build \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    AZUREML_MODEL_DIR=/var/azureml-model \
    MII_MODEL_DIR=/var/azureml-model \
    AZUREML_ENTRY_SCRIPT=score.py \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    MII_CACHE_PATH=/tmp/mii_cache

COPY . $BUILD_DIR

RUN mkdir -p $BUILD_DIR && \
    apt-get update && \
    apt-get install -y --no-install-recommends nginx-light wget sudo runit rsyslog libcurl4 unzip git-all && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /usr/share/man/* /var/lib/apt/lists/* && \
    mv "$BUILD_DIR/gunicorn_app" /etc/nginx/sites-available/ && \
    rm /etc/nginx/sites-enabled/default && \
    ln -s /etc/nginx/sites-available/gunicorn_app /etc/nginx/sites-enabled/ && \
    useradd --create-home dockeruser && \
    usermod -aG sudo dockeruser && \
    echo "dockeruser ALL=(ALL:ALL) NOPASSWD:/usr/sbin/service nginx start" >> /etc/sudoers.d/dockeruser && \
    mkdir -p /opt/miniconda /var/azureml-logger /var/azureml-util  && \
    chown -R dockeruser:root /opt/miniconda && \
    cp -r "$BUILD_DIR/runit" /var && \
    chown -R dockeruser:root /var/runit  && \
    mkdir -p {$AZUREML_MODEL_DIR,$MII_CACHE_PATH} && chmod 775 {$AZUREML_MODEL_DIR,$MII_CACHE_PATH} && chown -R dockeruser:root {$AZUREML_MODEL_DIR,$MII_CACHE_PATH}

ENV PATH=/opt/miniconda/envs/amlenv/bin:/opt/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    AZUREML_CONDA_ENVIRONMENT_PATH=/opt/miniconda/envs/amlenv  \
    LD_LIBRARY_PATH=/usr/local/:/usr/local/lib:/usr/local/cuda:/usr/local/nvidia/lib:$LD_LIBRARY_PATH \
    SVDIR=/var/runit \
    AZUREML_INFERENCE_SERVER_HTTP_ENABLED=True

USER dockeruser

SHELL ["/bin/bash", "-c"]

RUN cd ~ && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -bf -p /opt/miniconda && \
    conda create -n amlenv python=3.10 -y

ENV PATH="/opt/miniconda/envs/amlenv/bin:$AML_APP_ROOT:$PATH" \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH="/opt/miniconda/envs/amlenv/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" \
    CONDA_DEFAULT_ENV=amlenv \
    PATH=$PATH:/usr/local/cuda/bin

RUN /opt/miniconda/envs/amlenv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113 && \
    /opt/miniconda/envs/amlenv/bin/pip install -r "$BUILD_DIR/requirements.txt" && \
    /opt/miniconda/envs/amlenv/bin/pip install azureml-inference-server-http && \
    /opt/miniconda/envs/amlenv/bin/pip install git+https://github.com/microsoft/DeepSpeed.git && \
    /opt/miniconda/envs/amlenv/bin/pip install git+https://github.com/microsoft/DeepSpeed-MII.git && \
    /opt/miniconda/envs/amlenv/bin/pip install git+https://github.com/huggingface/transformers.git


EXPOSE 5001

WORKDIR $AZUREML_MODEL_DIR/code

CMD sudo service nginx start && cd $AZUREML_MODEL_DIR/code && azmlinfsrv --model_dir $AZUREML_MODEL_DIR --entry_script $AZUREML_MODEL_DIR/code/score.py --port 31311
"""

gunicorn = \
"""upstream gunicorn {
    server 127.0.0.1:31311;
}

server {
listen *:5001;
    location / {
        proxy_set_header Host $http_host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_redirect off;
        proxy_buffering off;
        proxy_pass http://gunicorn;
    }
}

  map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
  }
"""

gunicorn_run = \
"""#!/bin/bash


SCRIPT_PATH=$(dirname $(realpath -s "$0"))

# Error handling that sleeps so logs are properly sent
handle_error () {
  echo "Error occurred. Sleeping to send error logs."
  # Sleep 45 seconds
  sleep 45
  exit 95
}

format_print () {
    echo "$(date -uIns) | gunicorn/run | $1"
}

echo "`date -uIns` - gunicorn/run $@"

format_print ""
format_print "###############################################"
format_print "AzureML Container Runtime Information"
format_print "###############################################"
format_print ""


if [[ -z "${AZUREML_CONDA_ENVIRONMENT_PATH}" ]]; then
    # If AZUREML_CONDA_ENVIRONMENT_PATH exists, add to the front of the LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$(conda info --root)/lib:$LD_LIBRARY_PATH"
else
    # Otherwise, take the conda root and add that to the front of the LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH"
fi

if [[ -f "/IMAGE_INFORMATION" ]]; then
    format_print ""
    format_print "AzureML image information: $(cat /IMAGE_INFORMATION)"
    format_print ""
fi

format_print ""
format_print "PATH environment variable: $PATH"
format_print "PYTHONPATH environment variable: $PYTHONPATH"
format_print ""
format_print "Pip Dependencies (before dynamic installation)"
echo
pip freeze
echo

if [[ -n "$AZUREML_INFERENCE_SERVER_HTTP_ENABLED" ]]; then
    # Currently locking this feature to inference images.

    if [[ -n "$AZUREML_ENTRY_SCRIPT" ]]; then
        # Remove leading forward slash if it exists and then append the directory to the AML_APP_ROOT
        export ENTRY_SCRIPT_DIR="${AML_APP_ROOT:-/var/azureml-app}/$(dirname "${AZUREML_ENTRY_SCRIPT#/}")"
    else
        export ENTRY_SCRIPT_DIR=${AML_APP_ROOT:-/var/azureml-app}
    fi

    format_print ""
    format_print "Entry script directory: $ENTRY_SCRIPT_DIR"
    format_print ""
    format_print "###############################################"
    format_print "Dynamic Python Package Installation"
    format_print "###############################################"
    format_print ""


    if [[ -n "$AZUREML_EXTRA_PYTHON_LIB_PATH" ]]; then
        # Pre-installed mounted dependencies, check for the variable and if the folder exists.

        export EXTRA_PYTHON_LIB_FULL_PATH="${ENTRY_SCRIPT_DIR}/${AZUREML_EXTRA_PYTHON_LIB_PATH}"

        if [[ -d $EXTRA_PYTHON_LIB_FULL_PATH ]]; then
            format_print "Adding ${EXTRA_PYTHON_LIB_FULL_PATH} in PYTHONPATH"
            export PYTHONPATH="${EXTRA_PYTHON_LIB_FULL_PATH}:$PYTHONPATH"
        else
            format_print "Expected folder with pre-installed packages not found: ${EXTRA_PYTHON_LIB_FULL_PATH}. Exiting with error ..."
            exit 97
        fi
    elif [[ -n "$AZUREML_EXTRA_CONDA_YAML_ABS_PATH" || -n "$AZUREML_EXTRA_CONDA_YAML" ]]; then
        # Dynamic installation conda.yml, check for the variable and if the file exists for relative and absolute paths.
        # Need the absolute path for the MLFlow scenario where yaml could exist outside of azureml-app folder.

        if [[ -n "$AZUREML_EXTRA_CONDA_YAML_ABS_PATH" ]]; then
            export CONDA_FULL_PATH="$AZUREML_EXTRA_CONDA_YAML_ABS_PATH"
        else
            export CONDA_FULL_PATH="${ENTRY_SCRIPT_DIR}/${AZUREML_EXTRA_CONDA_YAML}"
        fi

        # NOTE: This may take a very long time if existing dependencies are added!
        # Source: https://stackoverflow.com/questions/53250933/conda-takes-20-minutes-to-solve-environment-when-package-is-already-installed
        if [[ -f $CONDA_FULL_PATH ]]; then
            format_print "Updating conda environment from ${CONDA_FULL_PATH} !"

            # Extract version from amlenv
            # If this is not installed, the value is empty. There will be a Warning output that states that the package is not installed.
            SERVER_VERSION="$(pip show azureml-inference-server-http | grep Version | sed -e 's/.*: //')"

            if [ -z "$SERVER_VERSION" ]; then
                format_print "azureml-inference-server-http not installed"
                exit 96
            fi

            # Copy user conda.yml to tmp folder since we don't have write access to user folder
            # Write access to folder is required for conda env create, and tmp folder has write access
            export CONDA_FILENAME="${TMPDIR:=/tmp}/copied_env_$(date +%s%N).yaml"

            cp "${CONDA_FULL_PATH}" "${CONDA_FILENAME}"

            # Create a userenv from the conda yaml that replaces the existing amlenv
            conda env create -n userenv -f "${CONDA_FILENAME}" || { handle_error ; }

            export AZUREML_CONDA_ENVIRONMENT_PATH="/opt/miniconda/envs/userenv"
            export PATH="/opt/miniconda/envs/userenv/bin:$PATH"
            export LD_LIBRARY_PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH"

            # Install the same version of the http server
            pip install azureml-inference-server-http=="$SERVER_VERSION" || { handle_error ; }

        else
            format_print "Dynamic Python packages installation is enabled but expected conda yaml file not found: ${CONDA_FULL_PATH}. Exiting with error ..."
            exit 98
        fi
    elif [[ -n "$AZUREML_EXTRA_REQUIREMENTS_TXT" ]]; then
        # Dynamic installation requirements.txt, check for the variable and if the file exists for relative and absolute paths.

        export REQUIREMENTS_TXT_FULL_PATH="${ENTRY_SCRIPT_DIR}/${AZUREML_EXTRA_REQUIREMENTS_TXT}"

        if [[ -f $REQUIREMENTS_TXT_FULL_PATH ]]; then
            format_print "Installing Python packages from ${REQUIREMENTS_TXT_FULL_PATH} !"
            pip install -r "$REQUIREMENTS_TXT_FULL_PATH" || { handle_error ; }
        else
            format_print "Dynamic Python packages installation is enabled but expected requirements file not found: ${REQUIREMENTS_TXT_FULL_PATH}. Exiting with error ..."
            exit 99
        fi
    else
        format_print "Dynamic Python package installation is disabled."
    fi
fi

format_print ""
format_print "###############################################"
format_print "AzureML Inference Server"
format_print "###############################################"
format_print ""

cd "${AML_APP_ROOT:-/var/azureml-app}"

# Check the result of $(pip show ...) instead of $(which azmlinfsrv). If we launch azmlinfsrv we need to make sure it is
# from the active python environment. $(which azmlinfsrv) may point to the azmlinfsrv in a different virtual env.
if [[ -n "$AZUREML_INFERENCE_SERVER_HTTP_ENABLED" || -n "$(pip show azureml-inference-server-http 2>/dev/null)" ]]; then
    format_print "Starting AzureML Inference Server HTTP."

    # Ensure the presence of debugpy if the user enabled local debugging. See ensure_debugpy.py for more details.
    if [[ -n $AZUREML_DEBUG_PORT ]]; then
        python $SCRIPT_PATH/ensure_debugpy.py
        if [[ $? -ne 0 ]]; then
            format_print "Exiting because debugpy cannot be not injected into entry.py."
            exit 94
        fi
    fi

    exec azmlinfsrv --entry_script "${AZUREML_ENTRY_SCRIPT:-main.py}" --port 31311
else
    format_print ""
    format_print "Starting HTTP server"
    format_print ""

    export PYTHONPATH="${AML_SERVER_ROOT:-/var/azureml-server}:$PYTHONPATH"
    exec gunicorn -c "${AML_SERVER_ROOT:-/var/azureml-server}/gunicorn_conf.py" "entry:app"
fi
"""

gunicorn_finish = \
"""#!/bin/bash

exit_code="$1" # The exit code from gunicorn
signal="$2"    # The signal which caused gunicorn to exit (or 0)

echo "`date -uIns` - gunicorn/finish $@"
echo "`date -uIns` - Exit code $exit_code is not normal. Killing image."

killall -SIGHUP runsvdir
"""

requirements = \
"""torch>=2.0.0
grpcio
grpcio-tools
pydantic
asyncio
"""
