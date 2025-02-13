# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time

import requests


def test_assistants_http(args, agent_config=None):
    proxies = {"http": ""}
    url = f"http://{args.ip_addr}:{args.ext_port}/v1"

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

    query = {
        "agent_config": agent_config,
    }

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
    def add_message_and_run(user_message):
        query = {"role": "user", "content": user_message, "assistant_id": assistant_id}
        if ret := process_request(f"threads/{thread_id}/messages", query):
            pass
        else:
            print("Error when add messages !!!!")
            return

        print("You may cancel the running process with cmdline")
        print(f"curl {url}/threads/{thread_id}/runs/cancel -X POST -H 'Content-Type: application/json'")

        query = {"assistant_id": assistant_id}
        process_request(f"threads/{thread_id}/runs", query, is_stream=True)

    # step 4. First turn
    user_message = "Hi! I'm Bob."
    add_message_and_run(user_message)
    time.sleep(1)

    # step 5. Second turn
    user_message = "What is OPEA?"
    add_message_and_run(user_message)
    time.sleep(1)

    # step 6. Third turn
    user_message = "What is my name?"
    add_message_and_run(user_message)
    time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="react_llama")
    parser.add_argument("--ip_addr", type=str, default="127.0.0.1", help="endpoint ip address")
    parser.add_argument("--ext_port", type=str, default="9090", help="endpoint port")
    parser.add_argument("--llm_endpoint_url", type=str, default="http://localhost:8086", help="tgi/vllm endpoint")

    args, _ = parser.parse_known_args()

    agent_config = {
        "strategy": "react_llama",
        "stream": True,
        "llm_engine": "vllm",
        "llm_endpoint_url": args.llm_endpoint_url,
        "tools": "/home/user/comps/agent/src/tools/custom_tools.yaml",
        "with_memory": True,
        "memory_type": "store",
        "store_config": {"redis_uri": f"redis://{args.ip_addr}:6379"},
    }

    print("test args:", args)
    test_assistants_http(args, agent_config)
