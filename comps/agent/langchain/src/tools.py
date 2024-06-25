# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import importlib
import os
import sys

import yaml

# from pydantic import create_model, Field
from langchain.pydantic_v1 import BaseModel, Field, create_model
from langchain.tools import StructuredTool, BaseTool
from langchain_community.agent_toolkits.load_tools import load_tools


def generate_request_function(url):
    def process_request(query):
        import json

        import requests

        content = json.dumps({"query": query})
        print(content)
        try:
            resp = requests.post(url=url, data=content)
            ret = resp.text
            resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
        except requests.exceptions.RequestException as e:
            ret = f"An error occurred:{e}"
        print(ret)
        return ret

    return process_request


def load_func_str(func_str):
    # case 1: func is an endpoint api
    if func_str.startswith("http://") or func_str.startswith("https://"):
        return generate_request_function(func_str)

    # case 2: func is a python file + function
    elif ".py:" in func_str:
        file_path, func_name = func_str.rsplit(":", 1)
        file_name = os.path.basename(file_path).split(".")[0]
        spec = importlib.util.spec_from_file_location(file_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func_str = getattr(module, func_name)

    # case 3: func is a langchain tool
    elif not '.' in func_str:
        try:
            return load_tools([func_str])[0]
        except:
            pass

    # case 4: func is a python loadable module
    else:
        module_path, func_name = func_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func_str = getattr(module, func_name)
        try:
            tool_inst = func_str()
            if isinstance(tool_inst, BaseTool):
                return tool_inst
        except:
            pass
    return func_str


def load_func_args(tool_name, args_dict):
    fields = {}
    for arg_name, arg_item in args_dict.items():
        fields[arg_name] = (arg_item["type"], Field(description=arg_item["description"]))
    return create_model(f"{tool_name}Input", **fields, __base__=BaseModel)


def load_langchain_tool(tool_setting_tuple):
    tool_name = tool_setting_tuple[0]
    tool_setting = tool_setting_tuple[1]
    func_definition = load_func_str(tool_setting["callable_api"])
    if isinstance(func_definition, BaseTool):
        return func_definition
    else:
        func_inputs = load_func_args(tool_name, tool_setting["args_schema"])
        return StructuredTool(
            name=tool_name,
            description=tool_setting["description"],
            func=func_definition,
            args_schema=func_inputs,
        )


def load_yaml_tools(file_dir_path: str):
    tools_setting = yaml.safe_load(open(file_dir_path))
    tools = []
    for t in tools_setting.items():
        tools.append(load_langchain_tool(t))
    return tools


def load_python_tools(file_dir_path: str):
    spec = importlib.util.spec_from_file_location("custom_tools", file_dir_path)
    module = importlib.util.module_from_spec(spec)
    # sys.modules["custom_tools"] = module
    spec.loader.exec_module(module)
    return module.tools_descriptions()


def get_tools_descriptions(file_dir_path: str):
    tools = []
    file_path_list = []
    if os.path.isdir(file_dir_path):
        file_path_list += glob.glob(file_dir_path + "/*")
    else:
        file_path_list.append(file_dir_path)
    for file in file_path_list:
        if os.path.basename(file).endswith(".yaml"):
            tools += load_yaml_tools(file)
        elif os.path.basename(file).endswith(".yml"):
            tools += load_yaml_tools(file)
        elif os.path.basename(file).endswith(".py"):
            tools += load_python_tools(file)
        else:
            pass
    return tools
