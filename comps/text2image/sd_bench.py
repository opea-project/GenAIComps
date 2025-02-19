# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import concurrent.futures
import datetime
import os
import subprocess
import time

import numpy
import psutil
import requests
from loguru import logger

url = "http://192.168.123.103:9379/sdapi/v1/txt2img"

prompt_simple = "A beautiful photograph of Mt.Fuji during cherry blossom"
prompt_complex = "((a delicate big apple) ), made of diamond hung on branch in a beautiful dawn, in the background beautiful valleys, (Dew drops) , divine iridescent glowing, opalescent textures, volumetric light, ethereal, sparkling, light inside body, bioluminescence, studio photo, highly detailed, sharp focus, photorealism, 8k, best quality, ultra detail:1. 5, hyper detail, hdr, hyper detail, ((universe of stars inside the apple) )"
prompts = {"simple": prompt_simple, "complex": prompt_complex}
max_wait_time = 60 * 5
check_interval = 5

warmup_steps = 5

test_steps = 10


opea_data = {
    "prompt": prompt_simple,
    "num_images_per_prompt": 1,
    "width": 512,
    "height": 512,
    "num_inference_steps": 25,
}
# data = {"prompt":prompt_simple, "num_images_per_prompt":1,'width':1024,'height':1024, "num_inference_steps":25}
# data = {"prompt":prompt_simple, "num_images_per_prompt":1,'width':1024,'height':1024, "num_inference_steps":4}
# data = {"prompt":prompt_simple, "num_images_per_prompt":1,'width':512,'height':512, "num_inference_steps":25}

webui_data = {
    "batch_size": 1,
    "cfg_scale": 7,
    "denoising_strength": 0,
    "enable_hr": False,
    "eta": 0,
    "firstphase_height": 0,
    "firstphase_width": 0,
    "height": 512,
    "n_iter": 1,
    "negative_prompt": "",
    "prompt": "An astronaut riding a green horse",
    "restore_faces": False,
    "s_churn": 0,
    "s_noise": 1,
    "s_tmax": 0,
    "s_tmin": 0,
    "sampler_index": "Euler a",
    "seed": -1,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "steps": 25,
    "styles": [],
    "subseed": -1,
    "subseed_strength": 0,
    "tiling": False,
    "width": 512,
}

data = {}


def create_folder():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M")
    milliseconds = now.microsecond // 1000
    formatted_milliseconds = f"{milliseconds:03d}"
    folder_name = f"result_{formatted_time}_{formatted_milliseconds}"
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, "framework_bench", folder_name)
    print(folder_path)
    os.makedirs(folder_path)
    return folder_path


def send_single_request(idx, queries, concurrency, url, save=False):
    res = []
    headers = {"Content-Type": "application/json"}

    while idx < len(queries):
        start_time = time.time()
        response = requests.post(url, json=data, headers=headers)
        end_time = time.time()
        res.append({"idx": idx, "start": start_time, "end": end_time})
        idx += concurrency
        if save:
            images = response.json()["images"]
            img_path = create_folder()
            count = 0
            for image in images:
                image_data = base64.b64decode(image)

                with open(os.path.join(img_path, f"output_{count}.png"), "wb") as image_file:
                    image_file.write(image_data)
                count += 1

    return res


query = ""


def send_concurrency_requests(request_url, num_queries):
    concurrency = 1

    responses = []
    stock_queries = [query for _ in range(num_queries)]
    test_start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(concurrency):
            futures.append(
                executor.submit(
                    send_single_request,
                    idx=i,
                    queries=stock_queries,
                    concurrency=concurrency,
                    url=request_url,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            responses = responses + future.result()
    test_end_time = time.time()

    print("=======================")
    for r in responses:
        r["total_time"] = r["end"] - r["start"]
        print("query:", r["idx"], "    time taken:", r["total_time"])

    logger.info(f"======={mode}======={bs}=========")
    logger.info(f"Total Concurrency: {concurrency}")
    logger.info(f"Total Requests: {len(stock_queries)}")
    logger.info(f"Total Test time: {test_end_time - test_start_time}")

    response_times = [r["total_time"] for r in responses]

    avg = numpy.mean(response_times)

    logger.info(f"AVG latency is {avg}s")

    # Calculate the P50 (median)
    p50 = numpy.percentile(response_times, 50)
    logger.info("P50 total latency is " + str(p50) + " s")

    # Calculate the P90 (median)
    p90 = numpy.percentile(response_times, 90)
    logger.info("P90 total latency is " + str(p90) + " s")

    # Calculate the P99
    p99 = numpy.percentile(response_times, 99)
    logger.info("P99 total latency is " + str(p99) + " s")

    return p50, p99


def start_process(command):
    env = os.environ.copy()
    env["http_proxy"] = ""
    print(command)
    with open("process.log", "a+") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, text=True, env=env, shell=True)
    return process


def wait_for_service(target_url):
    start_time = time.time()
    print("Wait for service ready")
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.post(target_url)
            if response.status_code == 422:
                print("Service started")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(check_interval)
    print("Start service time out")
    return False


def kill_process_tree(process):
    try:
        pid = process.pid
        print(f"Killing process tree, root process PID: {pid}")

        parent = psutil.Process(pid)

        children = parent.children(recursive=True)
        for child in children:
            print(f"Killing child process: PID={child.pid}")
            child.kill()

        print(f"Killing parent process: PID={parent.pid}")
        parent.kill()
        process.wait()
        print("Process tree successfully killed.")
    except psutil.NoSuchProcess:
        print(f"Process does not exist: PID={pid}")
    except psutil.AccessDenied:
        print(f"Permission denied to kill process: PID={pid}")
    except Exception as e:
        print(f"Failed to kill process tree: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent client to send POST requests")
    parser.add_argument("--batch_size", type=int, default=1, help="Number images per prompt")
    parser.add_argument("--url", type=str, default="http://192.168.123.103:57861/sdapi/v1/txt2img", help="Service URL")
    parser.add_argument("--model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="SD Model")
    args = parser.parse_args()
    webui_url = args.url

    data["num_images_per_prompt"] = args.batch_size
    if args.model == "stable-diffusion-v1-5/stable-diffusion-v1-5":
        index = 1
    elif args.model == "stabilityai/stable-diffusion-2-1":
        index = 2
    elif args.model == "stabilityai/stable-diffusion-xl-base-1.0":
        index = 3
    elif args.model == "ByteDance/SDXL-Lightning":
        index = 4

    logger.add("framework.log", format="{message}")

    data = webui_data
    if index > 2:
        data["width"] = 1024
        data["height"] = 1024
    if index == 4:
        data["steps"] = 4

    suc = wait_for_service(webui_url)

    if suc:
        try:
            logger.info(f"Start Benchmark\nurl: {args.url} ; model: {args.model}")

            for mode in ["complex", "simple"]:
                for bs in [1, 2, 4, 8, 10]:
                    print(f"warm up {warmup_steps} times")
                    for i in range(warmup_steps):
                        save_img = False
                        if i == 0:
                            save_img = True
                        data["batch_size"] = bs
                        data["prompt"] = prompts[mode]
                        send_single_request(0, [1], 5, webui_url, save_img)
                    send_concurrency_requests(request_url=webui_url, num_queries=test_steps)
        except Exception as e:
            print(e)
