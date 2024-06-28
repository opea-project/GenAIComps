# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess

import jsonlines
import yaml


def read_yaml_file(file_path):
    with open(file_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
    if os.path.exists("result_ragas.jsonl"):
        os.remove("result_ragas.jsonl")
    script_path = "ragas_benchmark.sh"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    data = read_yaml_file(args.config)
    data = {k: [str(item) for item in v] if isinstance(v, list) else str(v) for k, v in data.items()}

    ground_truth_file = data["ground_truth_file"]
    use_openai_key = data["use_openai_key"]
    search_types = data["search_type"]
    ks = data["k"]
    fetch_ks = data["fetch_k"]
    score_thresholds = data["score_threshold"]
    top_ns = data["top_n"]
    temperatures = data["temperature"]
    top_ks = data["top_k"]
    top_ps = data["top_p"]
    repetition_penaltys = data["repetition_penalty"]

    for search_type in search_types:
        for k in ks:
            for fetch_k in fetch_ks:
                for score_threshold in score_thresholds:
                    for top_n in top_ns:
                        for temperature in temperatures:
                            for top_k in top_ks:
                                for top_p in top_ps:
                                    for repetition_penalty in repetition_penaltys:
                                        subprocess.run(
                                            [
                                                "bash",
                                                script_path,
                                                "--ground_truth_file=" + ground_truth_file,
                                                "--use_openai_key=" + str(use_openai_key),
                                                "--search_type=" + search_type,
                                                "--k=" + k,
                                                "--fetch_k=" + fetch_k,
                                                "--score_threshold=" + score_threshold,
                                                "--top_n=" + top_n,
                                                "--temperature=" + temperature,
                                                "--top_k=" + top_k,
                                                "--top_p=" + top_p,
                                                "--repetition_penalty=" + repetition_penalty,
                                            ],
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL,
                                        )
