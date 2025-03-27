# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

try:
    from qwen_agent.tools.python_executor import PythonExecutor
except ImportError as e:
    raise ImportError(
        'The dependency of PythonExecutor is not installed. '
        'Please install the required dependency by running: pip install "qwen-agent[python_executor]"'
    ) from e


# tool for unit test
def search_web(query: str) -> str:
    """Search the web knowledge for a given query."""
    ret_text = """
    The Linux Foundation AI & Data announced the Open Platform for Enterprise AI (OPEA) as its latest Sandbox Project.
    OPEA aims to accelerate secure, cost-effective generative AI (GenAI) deployments for businesses by driving interoperability across a diverse and heterogeneous ecosystem, starting with retrieval-augmented generation (RAG).
    """
    return ret_text


def search_weather(query: str) -> str:
    """Search the weather for a given query."""
    ret_text = """
    It's clear.
    """
    return ret_text


def python_executor(
    code: str,
    get_answer_from_stdout=True,
    get_answer_expr=None,
    get_answer_symbol=None,
    timeout_length=20
) -> list:
    """
    Execute the given python code and return the result of code execution.
    
    Args:
        code: A string contains code to execute, format should be like ```\n<code>\n```.
        get_answer_from_stdout: A bool variable to specify whether to get execution result from stdout.
        get_answer_expr: A string variable to specify the python expression for getting execution result.
        get_answer_symbol: A string variable to specify the name of global variable which contains execution result.
        timeout_length: A int variable to specify timeout length for code execution in seconds.
    Returns:
        A tuple of code execution result, format as (execution result string, execution status string).
    """
    executor = PythonExecutor({
        "get_answer_from_stdout":get_answer_from_stdout,
        "get_answer_expr":get_answer_expr,
        "get_answer_symbol":get_answer_symbol,
        "timeout_length":timeout_length
    })
    return executor.call(code)


if __name__ == "__main__":
    # python_executor get_answer_from_stdout test
    print(python_executor("```\nprint('a')\nprint('b')\n```"))
    # python_executor get_answer_expr test
    print(python_executor("```\nl=2;r=3\n```", get_answer_from_stdout=False, get_answer_expr="l*r"))
    # python_executor get_answer_symbol test
    print(python_executor("```\nvar=1\n```", get_answer_from_stdout=False, get_answer_symbol="var"))