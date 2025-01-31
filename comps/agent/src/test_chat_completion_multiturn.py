# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import uuid

import requests


def process_request(url, query, is_stream=False):
    proxies = {"http": ""}
    content = json.dumps(query) if query is not None else None
    try:
        resp = requests.post(url=url, data=content, proxies=proxies, stream=is_stream)
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


def add_message_and_run(url, user_message, thread_id, stream=False):
    query = {"role": "user", "messages": user_message, "thread_id": thread_id, "stream": stream}
    ret = process_request(url, query, is_stream=stream)
    print("Response: ", ret)


def test_chat_completion_http(args):
    url = f"http://{args.ip_addr}:{args.ext_port}/v1/chat/completions"
    thread_id = f"{uuid.uuid4()}"

    # first turn
    user_message = "Hi! I'm Bob."
    add_message_and_run(url, user_message, thread_id, stream=args.stream)

    # second turn
    user_message = "What's OPEA project?"
    add_message_and_run(url, user_message, thread_id, stream=args.stream)

    # third turn
    user_message = "What is my name?"
    add_message_and_run(url, user_message, thread_id, stream=args.stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip_addr", type=str, default="127.0.0.1", help="endpoint ip address")
    parser.add_argument("--ext_port", type=str, default="9090", help="endpoint port")
    parser.add_argument("--llm_endpoint_url", type=str, default="http://localhost:8086", help="tgi/vllm endpoint")
    parser.add_argument("--stream", action="store_true", help="streaming mode")
    args, _ = parser.parse_known_args()

    print(args)
    test_chat_completion_http(args)
