# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import re

import jsonlines
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=E0401

from .prompt_dict import TRUTHGENERATE_PROMPT


def load_documents(document_file_jsonl_path):
    document_list = []
    with open(document_file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            passages = [stu["query"], stu["pos"][0]]
            document_list.append(passages)
    return document_list


def answer_generate(llm, base_dir, file_json_path, generation_config):
    documents = load_documents(base_dir)

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

    for question, context in enumerate(documents):
        if context and question:
            prompt = TRUTHGENERATE_PROMPT.format(question=question, context=context)
            if not use_endpoint:
                with torch.no_grad():
                    model_input = tokenizer(prompt, return_tensors="pt")
                    res = llm.generate(**model_input, generation_config=generation_config)[0]
                    res = tokenizer.decode(res, skip_special_tokens=True)
            else:
                res = llm.invoke(prompt)

            res = res[res.find("Generated ground_truth:") :]
            res = re.sub("Generated ground_truth:", "", res)
            res = re.sub("---", "", res)

            result_str = res.replace("#", " ").replace(r"\t", " ").replace("\n", " ").replace("\n\n", " ").strip()

            if result_str and not result_str.isspace():
                data = {
                    "question": question,
                    "context": [context],
                    "ground_truth": result_str,
                }
            with jsonlines.open(file_json_path, "a") as file_json:
                file_json.write(data)
