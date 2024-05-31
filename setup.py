#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#

import subprocess

from setuptools import find_packages, setup


def parse_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


setup(
    name="opea-eval",
    version="0.6",
    author="Intel AISE AIPC Team",
    author_email="haihao.shen@intel.com, feng.tian@intel.com, chang1.wang@intel.com, kaokao.lv@intel.com",
    description="Evaluation and benchmark for Generative AI",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/opea-project/GenAIEval",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.10",
)
