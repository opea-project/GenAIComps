# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import uuid

from huggingface_hub import ChatCompletionOutputFunctionDefinition, ChatCompletionOutputToolCall
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.output_parsers import BaseOutputParser


class ReActLlamaOutputParser(BaseOutputParser):
    def parse(self, text: str):
        print("raw output from llm: ", text)
        json_lines = text.split("\n")
        output = []
        for line in json_lines:
            try:
                if "TOOL CALL:" in line:
                    line = line.replace("TOOL CALL:", "")
                if "FINAL ANSWER:" in line:
                    line = line.replace("FINAL ANSWER:", "")
                if "assistant" in line:
                    line = line.replace("assistant", "")
                parsed_line = json.loads(line)
                if isinstance(parsed_line, dict):
                    print("parsed line: ", parsed_line)
                    output.append(parsed_line)
            except Exception as e:
                print("Exception happened in output parsing: ", str(e))
        if output:
            return output
        else:
            return None


def convert_json_to_tool_call(json_str):
    tool_name = json_str["tool"]
    tool_args = json_str["args"]
    tcid = str(uuid.uuid4())
    # add_kw_tc = {
    #     "tool_calls": [
    #         ChatCompletionOutputToolCall(
    #             function=ChatCompletionOutputFunctionDefinition(arguments=tool_args, name=tool_name, description=None),
    #             id=tcid,
    #             type="function",
    #         )
    #     ]
    # }
    tool_call = ToolCall(name=tool_name, args=tool_args, id=tcid)
    return tool_call


def get_tool_output(messages, id):
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id == id:
                tool_output = msg.content
                break
    return tool_output


def assemble_history(messages):
    """
    messages: AI, TOOL, AI, TOOL, etc.
    """
    query_history = ""
    breaker = "-" * 10
    for m in messages[1:]:  # exclude the first message
        if isinstance(m, AIMessage):
            # if there is tool call
            if hasattr(m, "tool_calls") and len(m.tool_calls) > 0:
                for tool_call in m.tool_calls:
                    tool = tool_call["name"]
                    tc_args = tool_call["args"]
                    id = tool_call["id"]
                    tool_output = get_tool_output(messages, id)
                    query_history += f"Tool Call: {tool} - {tc_args}\nTool Output: {tool_output}\n{breaker}\n"
            else:
                # did not make tool calls
                query_history += f"Assistant Output: {m.content}\n"

    return query_history


def assemble_memory(messages):
    """
    messages: Human, AI, TOOL, AI, TOOL, etc. in a thread with multi-turn conversations
    output: 
    query - user input of current turn. 
    conversation_history - history user input and final ai output in previous turns.
    query_history - history of tool calls and outputs in current turn.

    How to disect turns: each human message signals the start of a new turn.
    """
    query = ""
    query_id = None
    query_history = ""
    breaker = "-" * 10

    # get most recent human input
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            query = m.content
            query_id = m.id
            most_recent_human_message_idx = messages.index(m)
            break

    # get query history in this turn
    # start from the most recent human input
    for m in messages[most_recent_human_message_idx:]:
        if isinstance(m, AIMessage):
            # if there is tool call
            if hasattr(m, "tool_calls") and len(m.tool_calls) > 0:
                for tool_call in m.tool_calls:
                    tool = tool_call["name"]
                    tc_args = tool_call["args"]
                    id = tool_call["id"]
                    tool_output = get_tool_output(messages, id)
                    query_history += f"Tool Call: {tool} - {tc_args}\nTool Output: {tool_output}\n{breaker}\n"
            else:
                # did not make tool calls
                query_history += f"Assistant Output: {m.content}\n"

        elif isinstance(m, HumanMessage):
            query_history += f"User Input: {m.content}\n"

    # get conversion history of previous turns
    conversation_history = ""
    for i, m in enumerate(messages[:most_recent_human_message_idx]):
        if isinstance(m, HumanMessage):
            conversation_history += f"User Input: {m.content}\n"
        elif isinstance(m, AIMessage) and isinstance(messages[i + 1], HumanMessage):
            conversation_history += f"Assistant Output: {m.content}\n"
    return query, query_history, conversation_history


def save_state_to_store(state, config, store):

    # Get the user id from the config
    assistant_id = config["configurable"]["user_id"]
    thread_id = config["configurable"]["thread_id"]

    # Namespace the memory
    namespace = (assistant_id, thread_id) # pass in instead?

    # ... Analyze conversation and create a new memory

    # Create a new memory ID
    memory_id = str(uuid.uuid4())

    # We create a new memory
    store.put(namespace, memory_id, {"state": state})