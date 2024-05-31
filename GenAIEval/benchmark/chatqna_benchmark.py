# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import concurrent.futures
import json
import random
import time

import numpy
import requests

response_times = []


def extract_qText(json_data):
    try:
        file = open("data.json")
        data = json.load(file)
        post_json_data = {}
        post_json_data["model"] = "Intel/neural-chat-7b-v3-3"
        post_json_data["messages"] = data[random.randint(0, len(data) - 1)]["qText"]
        return json.dumps(post_json_data)
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def send_request(url, json_data):
    global response_times
    print(f"Sending request to {url} with data {json_data}")
    start_time = time.time()
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json_data, headers=headers)
    end_time = time.time()
    response_times.append(end_time - start_time)
    print(f"Question: {json_data} Response: {response.status_code} - {response.text}")


def main(url, json_data, concurrency):
    global response_times
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_url = {
            executor.submit(send_request, url, extract_qText(json_data)): url for _ in range(concurrency * 2)
        }
        for future in concurrent.futures.as_completed(future_to_url):
            _ = future_to_url[future]

    print(f"Total Requests: {concurrency*2}")

    # Calculate the P50 (median)
    p50 = numpy.percentile(response_times, 50)
    print("P50 latency is ", p50, "s")

    # Calculate the P99
    p99 = numpy.percentile(response_times, 99)
    print("P99 latency is ", p99, "s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent client to send POST requests")
    parser.add_argument(
        "--backend_url", type=str, default="http://localhost:8888/v1/chatqna", help="Service URL to send requests to"
    )
    parser.add_argument(
        "--json_data",
        type=str,
        default='{"inputs":"Which NFL team won the Super Bowl in the 2010 season?","parameters":{"do_sample": true}}',
        help="JSON data to send",
    )
    parser.add_argument("--concurrency", type=int, default=100, help="Concurrency level")
    args = parser.parse_args()
    main(args.backend_url, args.json_data, args.concurrency)
