# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from langchain_core.messages import AIMessage, ToolMessage

def assemble_history(messages):
    """
    messages: AI, TOOL, AI, TOOL, etc.
    """
    query_history = ""
    n = 1
    for m in messages[1:]:  # exclude the first message
        if isinstance(m, AIMessage):
            # if there is tool call
            if hasattr(m, "tool_calls") and len(m.tool_calls) > 0:
                for tool_call in m.tool_calls:
                    tool = tool_call["name"]
                    tc_args = tool_call["args"]
                    query_history += f"Tool Call: {tool} - {tc_args}\n"
            else:
                # did not make tool calls
                query_history += f"Assistant Output {n}: {m.content}\n"
        elif isinstance(m, ToolMessage):
            query_history += f"Tool Output: {m.content}\n"
    return query_history
