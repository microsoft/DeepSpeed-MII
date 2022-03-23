'''
Copyright 2022 The Microsoft DeepSpeed Team
'''

from setuptools import setup, find_packages


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


install_requires = fetch_requirements('requirements/requirements.txt')

extras_require = {"dev": fetch_requirements('requirements/requirements-dev.txt')}

setup(
    name='mii',
    version="0.1.0",
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
