# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from huggingface_hub import ChatCompletionOutputFunctionDefinition, ChatCompletionOutputToolCall
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel


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
    Assemble memory from messages within this thread (i.e., same thread id)
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


class ToolCallObject(BaseModel):
    name: str
    args: Dict[str, Any]
    id: str


class StoreMessage(BaseModel):
    id: str
    object: str = "thread.message"
    created_at: float
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallObject]] = None
    tool_call_id: Optional[str] = None


def convert_to_message_object(message):
    if isinstance(message, HumanMessage):
        message_object = StoreMessage(
            id=message.id,
            created_at=time.time(),
            role="user",
            content=message.content,
        )
    elif isinstance(message, AIMessage):
        if message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                tool_calls.append(
                    {
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                        "id": tool_call["id"],
                    }
                )
        else:
            tool_calls = None

        message_object = StoreMessage(
            id=message.id,
            created_at=time.time(),
            role="assistant",
            content=message.content,
            tool_calls=tool_calls,
        )

    elif isinstance(message, ToolMessage):
        message_object = StoreMessage(
            id=message.id,
            created_at=time.time(),
            role="tool",
            content=message.content,
            tool_call_id=message.tool_call_id,
        )
    else:
        raise ValueError("Invalid message type")

    return message_object


def save_state_to_store(state, config, store):
    last_message = state["messages"][-1]

    assistant_id = config["configurable"]["user_id"]
    thread_id = config["configurable"]["thread_id"]
    namespace = f"{assistant_id}_{thread_id}"

    # Create a new memory ID
    memory_id = str(uuid.uuid4())

    # convert message into MessageObject
    message_object = convert_to_message_object(last_message)
    store.put(memory_id, message_object.model_dump_json(), namespace)


def convert_from_message_object(message_object):
    if message_object["role"] == "user":
        try:
            # MessageObject class has a different structure from StoreMessage
            message = HumanMessage(content=message_object["content"][0]["text"], id=message_object["id"])
        except:
            message = HumanMessage(content=message_object["content"], id=message_object["id"])
    elif message_object["role"] == "assistant":
        if message_object["tool_calls"]:
            tool_calls = []
            for tool_call in message_object["tool_calls"]:
                tool_calls.append(ToolCall(name=tool_call["name"], args=tool_call["args"], id=tool_call["id"]))
            message = AIMessage(content=message_object["content"], id=message_object["id"], tool_calls=tool_calls)
        else:
            message = AIMessage(content=message_object["content"], id=message_object["id"])
    elif message_object["role"] == "tool":
        message = ToolMessage(content=message_object["content"], tool_call_id=message_object["tool_call_id"])
    else:
        raise ValueError("Invalid message role")
    return message


def assemble_memory_from_store(config, store):
    """
    store: RedisPersistence
    """
    assistant_id = config["configurable"]["user_id"]
    thread_id = config["configurable"]["thread_id"]
    namespace = f"{assistant_id}_{thread_id}"
    print("@@@Namespace: ", namespace)

    # get all the messages in this thread
    saved_all = store.get_all(namespace)
    message_objects = []
    messages = []
    for saved in saved_all:
        message_object = json.loads(saved_all[saved])
        print("@@@@ Saved memory:\n", message_object)
        message_objects.append(message_object)

    message_objects = sorted(message_objects, key=lambda x: x["created_at"])

    for message_object in message_objects:
        message = convert_from_message_object(message_object)
        messages.append(message)

    # print("@@@@ All messages:\n", messages)

    query, query_history, conversation_history = assemble_memory(messages)
    return query, query_history, conversation_history


def convert_aimessage_to_chat_completion(response: Union[dict, Any], stream=False, metadata=None):
    """
    convert langchain output back to openai chat completion format
    https://api.python.langchain.com/en/latest/_modules/langchain_openai/chat_models/base.html#ChatOpenAI
    """
    if not stream:
        usage = response.response_metadata["token_usage"]
        chat_id = response.response_metadata["id"]
        model = response.response_metadata["model_name"]
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": response.content + "\n", "tool_calls": []},
            "logprobs": response.response_metadata["logprobs"],
            "finish_reason": response.response_metadata["finish_reason"],
            "stop_reason": None,
        }
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": "",
            "choices": [choice],
            "model": model,
            "usage": usage,
            "prompt_logprobs": None,
        }
    else:
        choice = {
            "index": 0,
            "delta": {"content": response.content},
            "logprobs": None,
            "finish_reason": None,
        }
        return {
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": "",
            "model": metadata.get("ls_model_name", None),
            "choices": [choice],
        }


def convert_think_to_chat_completion(think):
    choice = {
        "index": 0,
        "delta": {"content": think},
        "logprobs": None,
        "finish_reason": None,
    }
    return {
        "id": "",
        "object": "chat.completion.chunk",
        "created": "",
        "model": "",
        "choices": [choice],
    }
