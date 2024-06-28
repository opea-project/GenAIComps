# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import random

import jsonlines
import numpy
import requests


def similarity_score(queries, passages, model):
    queries = [queries]
    passages = passages
    instruction = ""
    q_embeddings = model.encode([instruction + q for q in queries], normalize_embeddings=True)
    p_embeddings = model.encode(passages, normalize_embeddings=True)
    similarity_score = q_embeddings @ p_embeddings.T
    return similarity_score


def similarity_check(file_jsonl_path, file_json_split_path, model, similarity_threshold):
    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            stu["query"] = stu["query"].split("?")[:-1]
            for query in stu["query"]:
                query = query.lstrip("0123456789-. ") + "?"
                if similarity_score(query, stu["pos"], model) >= similarity_threshold:
                    data = {
                        "query": query,
                        "pos": stu["pos"],
                        "neg": stu["neg"],
                    }
                    with jsonlines.open(file_json_split_path, "a") as file_json:
                        file_json.write(data)
