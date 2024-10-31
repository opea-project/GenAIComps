# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import requests


def generate_answer_agent_api(url, prompt):
    proxies = {"http": ""}
    payload = {
        "query": prompt,
    }
    response = requests.post(url, json=payload, proxies=proxies)
    answer = response.json()["text"]
    return answer


if __name__ == "__main__":
    ip_address = os.getenv("ip_address", "localhost")
    url = f"http://{ip_address}:9095/v1/chat/completions"
    prompt = "What is OPEA?"
    answer = generate_answer_agent_api(url, prompt)
    print(answer)
