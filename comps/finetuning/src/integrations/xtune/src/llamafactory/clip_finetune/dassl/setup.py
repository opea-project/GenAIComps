# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os.path as osp

import numpy as np
from setuptools import find_packages, setup


def find_version():
    version_file = "dassl/__init__.py"
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


def get_requirements(filename="requirements.txt"):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires


setup(
    name="dassl",
    version=find_version(),
    description="Dassl: Domain adaptation and semi-supervised learning",
    author="Kaiyang Zhou",
    license="MIT",
    url="https://github.com/KaiyangZhou/Dassl.pytorch",
    packages=find_packages(),
    install_requires=get_requirements(),
    keywords=["Domain Adaptation", "Domain Generalization", "Semi-Supervised Learning", "Pytorch"],
)
