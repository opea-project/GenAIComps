# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import traceback

import pandas as pd
import requests
from integrations.utils import get_args


def test_agent_local(args):
    from integrations.agent import instantiate_agent

    agent = instantiate_agent(args)

    config = {"recursion_limit": args.recursion_limit}

    query = "What is OPEA project?"

    # run_agent(agent, config, query)


def test_agent_http(args):
    proxies = {"http": ""}
    ip_addr = args.ip_addr
    url = f"http://{ip_addr}:9090/v1/chat/completions"

    def process_request(query):
        content = json.dumps({"query": query})
        print(content)
        try:
            resp = requests.post(url=url, data=content, proxies=proxies)
            ret = resp.text
            resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
        except requests.exceptions.RequestException as e:
            ret = f"An error occurred:{e}"
        print(ret)
        return ret

    if args.quick_test:
        df = pd.DataFrame({"query": ["What is the weather today in Austin?"]})
    elif args.quick_test_multi_args:
        df = pd.DataFrame({"query": ["what is the trade volume for Microsoft today?"]})
    else:
        df = pd.read_csv(os.path.join(args.filedir, args.filename))
        df = df.sample(n=2, random_state=42)
    traces = []
    for _, row in df.iterrows():
        ret = process_request(row["query"])
        trace = {"query": row["query"], "trace": ret}
        traces.append(trace)

    df["trace"] = traces
    df.to_csv(os.path.join(args.filedir, args.output), index=False)


def test_assistants_http(args):
    proxies = {"http": ""}
    ip_addr = args.ip_addr
    url = f"http://{ip_addr}:9090/v1"

    def process_request(api, query, is_stream=False):
        content = json.dumps(query) if query is not None else None
        print(f"send request to {url}/{api}, data is {content}")
        try:
            resp = requests.post(url=f"{url}/{api}", data=content, proxies=proxies, stream=is_stream)
            if not is_stream:
                ret = resp.json()
                print(ret)
            else:
                for line in resp.iter_lines(decode_unicode=True):
                    print(line)
                ret = None

            resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
            return ret
        except requests.exceptions.RequestException as e:
            ret = f"An error occurred:{e}"
            print(ret)
            return False

    # step 1. create assistants
    query = {}
    if ret := process_request("assistants", query):
        assistant_id = ret.get("id")
        print("Created Assistant Id: ", assistant_id)
    else:
        print("Error when creating assistants !!!!")
        return

    # step 2. create threads
    query = {}
    if ret := process_request("threads", query):
        thread_id = ret.get("id")
        print("Created Thread Id: ", thread_id)
    else:
        print("Error when creating threads !!!!")
        return

    # step 3. add messages
    if args.query is None:
        query = {"role": "user", "content": "How old was Bill Gates when he built Microsoft?"}
    else:
        query = {"role": "user", "content": args.query}
    if ret := process_request(f"threads/{thread_id}/messages", query):
        pass
    else:
        print("Error when add messages !!!!")
        return

    # step 4. run
    print("You may cancel the running process with cmdline")
    print(f"curl {url}/threads/{thread_id}/runs/cancel -X POST -H 'Content-Type: application/json'")

    query = {"assistant_id": assistant_id}
    process_request(f"threads/{thread_id}/runs", query, is_stream=True)


def test_ut(args):
    from integrations.tools import get_tools_descriptions

    tools = get_tools_descriptions("tools/custom_tools.py")
    for tool in tools:
        print(tool)


def run_agent(agent, config, input_message):
    initial_state = agent.prepare_initial_state(input_message)

    for s in agent.app.stream(initial_state, config=config, stream_mode="values"):
        message = s["messages"][-1]
        message.pretty_print()

    last_message = s["messages"][-1]
    print("******Response: ", last_message.content)


def stream_generator(agent, config, input_message):
    from integrations.strategy.react.utils import save_state_to_store

    initial_state = agent.prepare_initial_state(input_message)
    # try:
    for event in agent.app.stream(initial_state, config=config, stream_mode=["updates"]):
        print(event)
        event_type = event[0]
        data = event[1]
        if event_type == "updates":
            for node_name, node_state in data.items():
                print(f"@@@ {node_name} : {node_state}")
                print(" @@@ Save message to store....")
                save_state_to_store(node_state, config, agent.store)
                print(f"--- CALL {node_name} node ---\n")
                for k, v in node_state.items():
                    if v is not None:
                        print(f"------- {k}, {v} -------\n\n")
                        if node_name == "agent":
                            if v[0].content == "":
                                tool_names = []
                                for tool_call in v[0].tool_calls:
                                    tool_names.append(tool_call["name"])
                                result = {"tool": tool_names}
                            else:
                                result = {"content": [v[0].content.replace("\n\n", "\n")]}
                            # ui needs this format
                            print(f"data: {json.dumps(result)}\n\n")
                        elif node_name == "tools":
                            full_content = v[0].content
                            tool_name = v[0].name
                            result = {"tool": tool_name, "content": [full_content]}
                            print(f"data: {json.dumps(result)}\n\n")
                            if not full_content:
                                continue

    print("data: [DONE]\n\n")
    # except Exception as e:
    #     print(str(e))


import time
from uuid import uuid4


def save_message_to_store(db_client, namespace, input_message):
    msg_id = str(uuid4())
    input_object = json.dumps({"role": "user", "content": input_message, "id": msg_id, "created_at": int(time.time())})
    db_client.put(msg_id, input_object, namespace)


def test_memory(args):
    from integrations.agent import instantiate_agent

    agent = instantiate_agent(args)
    print(args)

    assistant_id = "my_assistant"
    thread_id = str(uuid4())
    namespace = f"{assistant_id}_{thread_id}"
    db_client = agent.store

    config = {
        "recursion_limit": 5,
        "configurable": {"session_id": thread_id, "thread_id": thread_id, "user_id": assistant_id},
    }

    input_message = "Hi! I'm Bob."
    save_message_to_store(db_client, namespace, input_message)
    run_agent(agent, config, input_message)
    time.sleep(1)
    print("============== End of first turn ==============")

    input_message = "What's OPEA project?"
    save_message_to_store(db_client, namespace, input_message)
    run_agent(agent, config, input_message)
    time.sleep(1)
    print("============== End of second turn ==============")

    input_message = "what's my name?"
    save_message_to_store(db_client, namespace, input_message)
    run_agent(agent, config, input_message)
    time.sleep(1)
    print("============== End of third turn ==============")

    # input_message = "Hi! I'm Bob."
    # msg_id = str(uuid4())
    # input_object = json.dumps({"role": "user", "content": input_message, "id": msg_id, "created_at": int(time.time())})
    # db_client.put(msg_id, input_object, namespace)
    # stream_generator(agent, config, input_message)
    # print("============== End of first turn ==============")

    # time.sleep(1)
    # input_message = "What's OPEA project?"
    # msg_id = str(uuid4())
    # input_object = json.dumps({"role": "user", "content": input_message, "id": msg_id, "created_at": int(time.time())})
    # db_client.put(msg_id, input_object, namespace)
    # stream_generator(agent, config, input_message)
    # print("============== End of second turn ==============")

    # time.sleep(1)
    # input_message = "what's my name?"
    # msg_id = str(uuid4())
    # input_object = json.dumps({"role": "user", "content": input_message, "id": msg_id, "created_at": int(time.time())})
    # db_client.put(msg_id, input_object, namespace)
    # stream_generator(agent, config, input_message)
    # print("============== End of third turn ==============")


if __name__ == "__main__":
    args1, _ = get_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="react_llama")
    parser.add_argument("--local_test", action="store_true", help="Test with local mode")
    parser.add_argument("--endpoint_test", action="store_true", help="Test with endpoint mode")
    parser.add_argument("--assistants_api_test", action="store_true", help="Test with endpoint mode")
    parser.add_argument("--q", type=int, default=0)
    parser.add_argument("--ip_addr", type=str, default="127.0.0.1", help="endpoint ip address")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--filedir", type=str, default="./", help="test file directory")
    parser.add_argument("--filename", type=str, default="query.csv", help="query_list_file")
    parser.add_argument("--output", type=str, default="output.csv", help="query_list_file")
    parser.add_argument("--ut", action="store_true", help="ut")

    args, _ = parser.parse_known_args()

    for key, value in vars(args1).items():
        setattr(args, key, value)

    # if args.local_test:
    #     test_agent_local(args)
    # elif args.endpoint_test:
    #     test_agent_http(args)
    # elif args.ut:
    #     test_ut(args)
    # elif args.assistants_api_test:
    #     test_assistants_http(args)
    # else:
    #     print("Please specify the test type")

    # test_memory(args)
    test_agent_local(args)
