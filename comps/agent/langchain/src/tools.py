# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import importlib
import os
import pathlib
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


# class ToolDescription:
#     name: str
#     description: str
#     func: callable
#     error_msg: str

# def load_tool(config):
#     tool = ToolDescription()
#     if isinstance(config, dict):
#         tool.name = config['name']
#         tool.description = config['description']
#         tool.func = config['func']
#     else:
#         tool.error_msg = config
#     return tool

# def load_tools(config_path):
#     tool_list = []
#     try:
#         with open(config_path, 'r') as stream:
#             tools_config = yaml.safe_load(stream)
#     except Exception as e:
#         tools_config = str(e)

#         if not isinstance(tools_config, list):
#             tools_config = [tools_config]
#         for cfg in tools_config:
#             tool = load_tool(cfg)
#             tool_list.append(tool)


# current_apth = pathlib.Path(__file__).parent.resolve()
# tools_path = os.path.join(current_apth, "../tools")
# tool_dir_list = os.listdir(tools_path)

# tool_list = []
# for tool in tool_dir_list:
#     tool_path = os.path.join(tools_path, tool)
#     config = os.path.join(tool_path, "config.yml")
#     with open(config, 'r') as stream:
#         tool_list.append(load_tool(yaml.safe_load(stream)))


# tools = os.listdir(os.path.dirname(__file__))

# def new_instance(module, clazz, **clazz_kwargs):
#     import importlib
#     module = importlib.import_module(module)
#     class_ = getattr(module, clazz)
#     return class_(**clazz_kwargs)
