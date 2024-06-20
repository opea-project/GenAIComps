# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import importlib
import os
import sys

import yaml


def get_tools_descriptions(file_dir_path: str):
    tools = []
    if os.path.isdir(file_dir_path):
        for file in glob.glob(file_dir_path):
            with open(file, "r") as stream:
                tools.append(yaml.safe_load(stream))
        return tools
    else:
        spec = importlib.util.spec_from_file_location("custom_tools", file_dir_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_tools"] = module
        spec.loader.exec_module(module)
        return module.tools_descriptions()
