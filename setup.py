# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import sys
import subprocess
from setuptools import setup, find_packages


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


install_requires = fetch_requirements('requirements/requirements.txt')

extras_require = {"dev": fetch_requirements('requirements/requirements-dev.txt')}


def command_exists(cmd):
    if sys.platform == "win32":
        result = subprocess.Popen(f'{cmd}', stdout=subprocess.PIPE, shell=True)
        return result.wait() == 1
    else:
        result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
        return result.wait() == 0


# Write out version/git info
git_hash_cmd = "git rev-parse --short HEAD"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
if command_exists('git') and 'DS_BUILD_STRING' not in os.environ:
    try:
        result = subprocess.check_output(git_hash_cmd, shell=True)
        git_hash = result.decode('utf-8').strip()
        result = subprocess.check_output(git_branch_cmd, shell=True)
        git_branch = result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"
        git_branch = "unknown"
else:
    git_hash = "unknown"
    git_branch = "unknown"

# Parse the MII version string from version.txt
version_str = open('version.txt', 'r').read().strip()

# Build specifiers like .devX can be added at install time. Otherwise, add the git hash.
# example: MII_BUILD_STR=".dev20201022" python setup.py sdist bdist_wheel

MII_BUILD_STRING = 'MII_BUILD_STRING'
BUILD_FILE = 'build.txt'
mii_build_string = os.environ.get(MII_BUILD_STRING)

# Building wheel for distribution, update version file
if mii_build_string:
    # Build string env specified, probably building for distribution
    with open(BUILD_FILE, 'w') as fd:
        fd.write(mii_build_string)
    version_str += mii_build_string
elif os.path.isfile(BUILD_FILE):
    # build.txt exists, probably installing from distribution
    with open(BUILD_FILE, 'r') as fd:
        version_str += fd.read().strip()
else:
    # None of the above, probably installing from source
    version_str += f'+{git_hash}'

# write out installed version
with open("mii/version.py", 'w') as fd:
    fd.write(f"__version__ = '{version_str}'\n")

# Parse README.md to make long_description for PyPI page.
thisdir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(thisdir, 'README.md'), encoding='utf-8') as fin:
    readme_text = fin.read()
print("PACKAGES", find_packages())
setup(name="deepspeed-mii",
      version=version_str,
      long_description=readme_text,
      long_description_content_type='text/markdown',
      description='deepspeed mii',
      author='DeepSpeed Team',
      author_email='deepspeed-mii@microsoft.com',
      url='http://deepspeed.ai',
      project_urls={
          'Documentation': 'https://github.com/microsoft/DeepSpeed-MII',
          'Source': 'https://github.com/microsoft/DeepSpeed-MII',
      },
      install_requires=install_requires,
      extras_require=extras_require,
      packages=find_packages(exclude=("tests",
                                      )),
      classifiers=[
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10'
      ])
