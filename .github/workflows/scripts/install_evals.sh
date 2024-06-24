#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

echo -e "\n Install GenAIEval ... "
cd /GenAIEval
python -m pip install --no-cache-dir -r requirements.txt
python setup.py bdist_wheel
pip install dist/opea_eval*.whl

pip list
