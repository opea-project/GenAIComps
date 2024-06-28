# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

from comps.dataprep.utils import document_loader
from langchain_community.llms import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig

from .gen_answer_dataset import answer_generate
from .gen_hard_negative import mine_hard_negatives
from .llm_generate_raw_data import raw_data_generation
from .utils import similarity_check

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str)
    parser.add_argument("--embedding_model", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str, default="./data")

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--num_return_sequences", type=int, default=2)
    parser.add_argument("--use_cache", type=bool, default=True)

    parser.add_argument("--range_for_sampling", type=str, default="2-10")
    parser.add_argument("--negative_number", type=int, default=5)
    parser.add_argument("--use_gpu_for_searching", type=bool, default=False)

    parser.add_argument("--similarity_threshold", type=float, default=0.6)

    args = parser.parse_args()

    llm_model = args.llm
    input_path = args.input
    output = args.output

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        use_cache=args.use_cache,
    )

    embedding_model = SentenceTransformer(args.embedding_model)

    try:
        llm_endpoint = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
        llm = HuggingFaceEndpoint(
            endpoint_url=llm_endpoint,
            max_new_tokens=512,
            top_k=args.top_k,
            top_p=args.top_p,
            typical_p=args.typical_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            streaming=args.streaming,
            timeout=600,
        )
    except:
        print("Did not find the llm endpoint service, load model from huggingface hub as instead.")

    try:
        if not os.path.exists(output):
            os.mkdir(output)
        else:
            if os.path.exists(os.path.join(output, "raw.jsonl")):
                os.remove(os.path.join(output, "raw.jsonl"))
            if os.path.exists(os.path.join(output, "minedHN.jsonl")):
                os.remove(os.path.join(output, "minedHN.jsonl"))
            if os.path.exists(os.path.join(output, "minedHN_split.jsonl")):
                os.remove(os.path.join(output, "minedHN_split.jsonl"))
    except:
        pass

    output_path = os.path.join(output, "raw_query.jsonl")
    raw_data_generation(llm, input_path, output_path, generation_config)

    output_hn_path = os.path.join(output, "query_doc.jsonl")
    mine_hard_negatives(
        embedding_model,
        output_path,
        output_hn_path,
        args.range_for_sampling,
        args.negative_number,
    )

    output_json_split_path = os.path.join(output, "query_doc_cleaned.jsonl")
    similarity_check(output_hn_path, output_json_split_path, embedding_model, args.similarity_threshold)

    output_answer_path = os.path.join(output, "answer.jsonl")
    answer_generate(llm, input, output, generation_config)
