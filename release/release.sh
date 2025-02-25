#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

cd ..

if [ ! -f ~/.pypirc ]; then
    echo 'create .pypirc in order to upload to PyPI'
    exit 1
fi

version=$1

if [ -z $version ]; then
    echo "please provide version number for release"
    exit 1
fi

if [[ $version == *"v"* ]]; then
    echo "please only include version number without 'v' prefix"
    exit 1
fi

if [ "${version}" != `cat version.txt` ]; then
    echo "version=${version} does not match version.txt"
    cat version.txt
    exit 1
fi

python -c "import twine"
if [ $? != 0 ]; then
    echo 'please install twine via pip'
    exit 1
fi

MII_BUILD_STRING="" python -m build --wheel
WHL=deepspeed_mii-${version}-py3-none-any.whl

if [ ! -f dist/${WHL} ]; then
    echo "prepared version does not match version given ($version), bump version first?"
    ls dist
    exit 1
fi

python -m twine upload --verbose dist/${WHL} --repository mii

git tag v${version}
git push origin v${version}

echo "bumping up patch version"
cd -
python bump_patch_version.py
