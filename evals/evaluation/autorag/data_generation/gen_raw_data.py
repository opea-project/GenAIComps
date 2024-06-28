# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from typing import List

import jsonlines
import torch
from comps.dataprep.utils import document_loader
from modelscope import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=E0401
from transformers import GenerationConfig

from .prompt_dict import QUERYGENERATE_PROMPT

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO
)


def document_filter(data_collection):
    documents = []
    for sample in data_collection:
        if len(sample) < 5:
            continue
        documents.append(sample)
    return documents


def raw_data_generation(llm, input_path, file_json_path, generation_config):
    data_collections = []

    if isinstance(input_path, str):
        if os.path.isfile(input_path):
            data_collection = document_loader(input_path)
            data_collections.append(data_collection)
        elif os.path.isdir(input_path):
            for dirpath, dirnames, filenames in os.walk(input_path):
                for filename in filenames:
                    data_collection = document_loader(os.path.join(dirpath, filename))
                    data_collections.append(data_collection)
    else:
        print("Please check your upload file and try again!")
    documents = document_filter(data_collection)

    try:
        if isinstance(llm, str):
            use_endpoint = False
            tokenizer = AutoTokenizer.from_pretrained(llm)
            llm = AutoModelForCausalLM.from_pretrained(llm, device_map="auto", torch_dtype=torch.float16)
            llm.eval()
        else:
            use_endpoint = True
            llm = llm
    except:
        print("Please check the setting llm!")

    for context in documents:
        if context:
            prompt = QUERYGENERATE_PROMPT.format(context=context)
            result = []

            for j in range(5):
                if not use_endpoint:
                    with torch.no_grad():
                        model_input = tokenizer(input, return_tensors="pt")
                        res = llm.generate(**model_input, generation_config=generation_config)[0]
                        res = tokenizer.decode(res, skip_special_tokens=True)
                else:
                    res = llm.invoke(prompt)

                res = res[res.find("Generated questions:") :]
                res = re.sub("Generated questions:", "", res)
                res = re.sub("---", "", res)
                res = res.split("?")[0:2]

                for content in res:
                    content = content.replace("1.", "").replace("2.", "")
                    content = content.replace("Evaluation:", "")
                    content = (
                        content.replace("#", " ").replace(r"\t", " ").replace("\n", " ").replace("\n\n", " ").strip()
                    )
                    content = content + "?"
                result.append(content)

            result_str = ""
            result_set = list(set(result))
            for k in range(len(result_set)):
                result_str = result_str + str(k) + ". " + result_set[k]

            if result_str and not result_str.isspace():
                data = {
                    "query": result_str,
                    "pos": [context],
                }
                with jsonlines.open(file_json_path, "a") as file_json:
                    file_json.write(data)
