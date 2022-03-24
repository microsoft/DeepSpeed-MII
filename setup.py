'''
Copyright 2022 The Microsoft DeepSpeed Team
'''

import os
import subprocess
from setuptools import setup, find_packages


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


install_requires = fetch_requirements('requirements/requirements.txt')

extras_require = {
    "dev": fetch_requirements('requirements/requirements-dev.txt'),
    "local": fetch_requirements('requirements/requirements-local.txt'),
    "azure": fetch_requirements('requirements/requirements-azure.txt')
}


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

# Building wheel for distribution, update version file
if 'MII_BUILD_STRING' in os.environ:
    # Build string env specified, probably building for distribution
    with open('build.txt', 'w') as fd:
        fd.write(os.environ.get('MII_BUILD_STRING'))
    version_str += os.environ.get('MII_BUILD_STRING')
elif os.path.isfile('build.txt'):
    # build.txt exists, probably installing from distribution
    with open('build.txt', 'r') as fd:
        version_str += fd.read().strip()
else:
    # None of the above, probably installing from source
    version_str += f'+{git_hash}'

setup(
    name='mii',
    version=version_str,
    description='deepspeed mii',
    install_requires=install_requires,
    #scripts=[],
    extras_require=extras_require,
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
        'Programming Language :: Python :: 3.10'
    ])
