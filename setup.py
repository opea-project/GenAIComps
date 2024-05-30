#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess

from setuptools import find_packages, setup


def parse_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


setup(
    name="GenAIEval",
    version="0.0.0",
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
