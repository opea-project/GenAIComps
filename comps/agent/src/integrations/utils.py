# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import json

from .config import env_config

LLM_ENDPOINT_URL_DEFAULT = "http://localhost:8080"


def format_date(date):
    # input m/dd/yyyy hr:min
    # output yyyy-mm-dd
    date = date.split(" ")[0]  # remove hr:min
    # print(date)
    try:
        date = date.split("/")  # list
        # print(date)
        year = date[2]
        month = date[0]
        if len(month) == 1:
            month = "0" + month
        day = date[1]
        return f"{year}-{month}-{day}"
    except:
        return date


def setup_hf_tgi_client(args):
    from langchain_huggingface import HuggingFaceEndpoint

    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "return_full_text": args.return_full_text,
        "streaming": args.stream,
    }

    llm = HuggingFaceEndpoint(
        endpoint_url=args.llm_endpoint_url,
        task="text-generation",
        **generation_params,
    )
    return llm


def setup_chat_model(args):
    from langchain_openai import ChatOpenAI

    params = {
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "streaming": args.stream,
    }
    if args.llm_engine == "vllm" or args.llm_engine == "tgi":
        openai_key = "EMPTY"
    elif args.llm_engine == "openai":
        openai_key = args.api_key
    else:
        raise ValueError("llm_engine must be vllm, tgi, or openai")

    openai_endpoint = None if args.llm_endpoint_url is LLM_ENDPOINT_URL_DEFAULT else args.llm_endpoint_url + "/v1"
    llm = ChatOpenAI(
        openai_api_key=openai_key,
        openai_api_base=openai_endpoint,
        model_name=args.model,
        request_timeout=args.timeout,
        **params,
    )
    return llm


def tool_renderer(tools):
    tool_strings = []
    for tool in tools:
        description = f"{tool.name} - {tool.description}"

        arg_schema = []
        for k, tool_dict in tool.args.items():
            k_type = tool_dict["type"] if "type" in tool_dict else ""
            k_desc = tool_dict["description"] if "description" in tool_dict else ""
            arg_schema.append(f"{k} ({k_type}): {k_desc}")

        tool_strings.append(f"{description}, args: {arg_schema}")
    return "\n".join(tool_strings)


def filter_tools(tools, tools_choices):
    tool_used = []
    for tool in tools:
        if tool.name in tools_choices:
            tool_used.append(tool)
    return tool_used


def has_multi_tool_inputs(tools):
    ret = False
    for tool in tools:
        if len(tool.args) > 1:
            ret = True
            break
    return ret


def load_python_prompt(file_dir_path: str):
    print(file_dir_path)
    spec = importlib.util.spec_from_file_location("custom_prompt", file_dir_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapt_custom_prompt(local_vars, custom_prompt):
    # list attributes of module
    if custom_prompt is not None:
        custom_prompt_list = [k for k in dir(custom_prompt) if k[:2] != "__"]
        for k in custom_prompt_list:
            v = getattr(custom_prompt, k)
            local_vars[k] = v


def assemble_store_messages(messages):

    inputs = []
    for mid in messages:
        message = json.loads(messages[mid])
        # TODO: format messages
        inputs.append("### " + message["role"].upper() + ":" + "\n" + message["content"][0]["text"])

    # revert messages
    return "\n".join(inputs)


def get_latest_human_message_from_store(store, namespace):
    messages = store.get_all(namespace)
    human_messages = []
    for mid in messages:
        message = json.loads(messages[mid])
        if message["role"] == "user":
            human_messages.append(message)

    human_messages = sorted(human_messages, key=lambda x: x["created_at"])
    latest_human_message = human_messages[-1]
    return latest_human_message["content"][0]["text"]


def get_args():
    parser = argparse.ArgumentParser()
    # llm args
    parser.add_argument("--stream", type=str, default="true")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--agent_name", type=str, default="OPEA_Default_Agent")
    parser.add_argument("--strategy", type=str, default="react_langchain")
    parser.add_argument("--role_description", type=str, default="LLM enhanced agent")
    parser.add_argument("--tools", type=str, default=None, help="path to the tools file")
    parser.add_argument("--mcp_sse_server_url", type=str, default=None, help="OPEA MCP SSE server URL")
    parser.add_argument("--mcp_sse_server_api_key", type=str, default=None, help="OPEA MCP SSE server API key")
    parser.add_argument("--recursion_limit", type=int, default=5)
    parser.add_argument("--require_human_feedback", action="store_true", help="If this agent requires human feedback")
    parser.add_argument("--debug", action="store_true", help="Test with endpoint mode")

    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--llm_engine", type=str, default="tgi")
    parser.add_argument("--llm_endpoint_url", type=str, default="http://localhost:8080")
    parser.add_argument("--api_key", type=str, default=None, help="API key to access remote server")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    parser.add_argument("--return_full_text", type=bool, default=False)
    parser.add_argument("--custom_prompt", type=str, default=None)
    parser.add_argument("--with_memory", type=str, default="true")
    parser.add_argument("--memory_type", type=str, default="checkpointer", help="choices: checkpointer, store")
    parser.add_argument("--timeout", type=int, default=60)

    # for sql agent
    parser.add_argument("--db_path", type=str, help="database path")
    parser.add_argument("--db_name", type=str, help="database name")
    parser.add_argument("--use_hints", type=str, default="false", help="If this agent uses hints")
    parser.add_argument("--hints_file", type=str, help="path to the hints file")

    sys_args, unknown_args = parser.parse_known_args()
    print("env_config: ", env_config)
    if env_config != []:
        env_args, env_unknown_args = parser.parse_known_args(env_config)
        unknown_args += env_unknown_args
        for key, value in vars(env_args).items():
            setattr(sys_args, key, value)

    if sys_args.stream == "true":
        sys_args.stream = True
    else:
        sys_args.stream = False

    if sys_args.with_memory == "true":
        sys_args.with_memory = True
    else:
        sys_args.with_memory = False

    if sys_args.use_hints == "true":
        print("SQL agent will use hints")
        sys_args.use_hints = True
    else:
        sys_args.use_hints = False

    print("==========sys_args==========:\n", sys_args)
    return sys_args, unknown_args
