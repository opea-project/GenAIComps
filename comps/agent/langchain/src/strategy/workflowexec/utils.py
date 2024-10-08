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
                pass
            else:
                # did not make tool calls
                query_history += f"Assistant Output {n}: {m.content}\n"
        elif isinstance(m, ToolMessage):
            query_history += f"Tool Output: {m.content}\n"
    return query_history

def prepare_tool_call(response, sender: str):
    tool_calls = []
    for tool_call in response.tool_calls:
        tool_calls.append(tool_call)

    if tool_calls:
        ai_message = AIMessage(content="", tool_calls=tool_calls)
    else:
        ai_message = AIMessage(content=response)
        
    return {"messages": [ai_message], "sender": sender}
