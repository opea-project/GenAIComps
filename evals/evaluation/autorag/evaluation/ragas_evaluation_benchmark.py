# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random
import shutil

import jsonlines
import numpy
import pandas as pd
import requests
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from ragas import evaluate  # pylint: disable=E0401
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness  # pylint: disable=E0401


def load_set(file_jsonl_path, item):
    list = []
    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            passages = stu[item]
            list.append(passages)
    return list


def rag_evaluate(
    backend_url,
    llm,
    ground_truth_file,
    use_openai_key,
    embedding_model,
    search_type,
    k,
    fetch_k,
    score_threshold,
    top_n,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
):
    result_answer_path = "./result_answer.jsonl"
    question_list = load_set(ground_truth_file, "question")
    for question in question_list:
        post_json_data = {}
        post_json_data["model"] = "Intel/neural-chat-7b-v3-3"
        post_json_data["messages"] = question
        post_json_data["search_type"] = search_type
        post_json_data["k"] = k
        post_json_data["fetch_k"] = fetch_k
        post_json_data["score_threshold"] = score_threshold
        post_json_data["top_n"] = top_n
        post_json_data["temperature"] = temperature
        post_json_data["top_k"] = top_k
        post_json_data["top_p"] = top_p
        post_json_data["repetition_penalty"] = repetition_penalty
        json_data = json.dumps(post_json_data)

        headers = {"Content-Type": "application/json"}
        response = requests.post(backend_url, data=json_data, headers=headers)
        rag_answer = json.loads(response.text)["choices"][0]["message"]["content"]

        data = {
            "question": question,
            "answer": rag_answer,
        }
        with jsonlines.open(result_answer_path, "a") as file_json:
            file_json.write(data)

    contexts_list = load_set(ground_truth_file, "context")
    ground_truth_list = load_set(ground_truth_file, "ground_truth")
    question_list = load_set(result_answer_path, "question")
    answer_list = load_set(result_answer_path, "answer")

    data_samples = {
        "question": question_list,
        "answer": answer_list,
        "contexts": contexts_list,
        "ground_truth": ground_truth_list,
    }

    dataset = Dataset.from_dict(data_samples)

    if use_openai_key:
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        score = evaluate(dataset, metrics=[answer_relevancy, faithfulness, context_recall, context_precision])
        df = score.to_pandas()
        answer_relevancy_average = df["answer_relevancy"][:].mean()
        faithfulness_average = df["faithfulness"][:].mean()
        context_recall_average = df["context_recall"][:].mean()
        context_precision_average = df["context_precision"][:].mean()
        print("The score for answer_relevancy is {}".format(answer_relevancy_average))
        print("The score for faithfulness is {}".format(faithfulness))
        print("The score for context_recall is {}".format(context_recall))
        print("The score for context_precision is {}".format(context_precision))
        print(
            """The current group of parameters is:
                search_type: %s, k: %d, fetch_k: %d, score_threshold: %f, top_n: %d, temperature: %f, \
                top_k: %d, top_p: %d, repetition_penalty: %f.
              """
            % (search_type, k, fetch_k, score_threshold, top_n, temperature, top_k, top_p, repetition_penalty)
        )
        return answer_relevancy_average, faithfulness_average, context_recall_average, context_precision_average
    else:
        try:
            if isinstance(llm, str):
                langchain_llm = HuggingFacePipeline.from_model_id(
                    model_id=llm,
                    task="text-generation",
                    pipeline_kwargs={"max_new_tokens": 128},
                )
            else:
                langchain_llm = llm
            langchain_llm = LangchainLLMWrapper(langchain_llm)

        except:
            print("Please check the selected llm!")

        langchain_embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages:",
        )

        langchain_embedding = LangchainEmbeddingsWrapper(langchain_embeddings)
        ### Note: due to the code error in RAGAS repo, do not recommend other model except chatgpt to
        # evaluate context_recall and context_precision.
        # Please refer https://github.com/explodinggradients/ragas/issues/664.
        score = evaluate(
            dataset,  # pylint: disable=E1123
            metrics=[answer_relevancy, faithfulness],
            llm=langchain_llm,  # pylint: disable=E1123
            embeddings=langchain_embedding,
        )  # pylint: disable=E1123

        df = score.to_pandas()
        answer_relevancy_average = df["answer_relevancy"][:].mean()
        faithfulness_average = df["faithfulness"][:].mean()
        print("The score for answer_relevancy is {}".format(answer_relevancy_average))
        print("The score for faithfulness is {}".format(faithfulness))
        print(
            """The current group of parameters is:
                search_type: %s, k: %d, fetch_k: %d, score_threshold: %f, top_n: %d, temperature: %f, \
                top_k: %d, top_p: %d, repetition_penalty: %f.
              """
            % (search_type, k, fetch_k, score_threshold, top_n, temperature, top_k, top_p, repetition_penalty)
        )
        return answer_relevancy_average, faithfulness_average


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend_url", type=str, default="http://localhost:8888/v1/chatqna", help="Service URL address."
    )
    parser.add_argument("--ground_truth_file", type=str)
    parser.add_argument("--use_openai_key", type=bool, default=False)

    parser.add_argument("--retrieval_type", type=str, default="default")
    parser.add_argument("--search_type", type=str, default="similarity")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--fetch_k", type=int, default=5)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--top_n", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()

    llm = "Intel/neural-chat-7b-v3-3"

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
    if args.use_openai_key:
        answer_relevancy_average, faithfulness_average, context_recall_average, context_precision_average = (
            rag_evaluate(
                backend_url=args.backend_url,
                llm=llm,
                ground_truth_file=args.ground_truth_file,
                use_openai_key=args.use_openai_key,
                embedding_model="BAAI/bge-large-en-v1.5",
                search_type=args.search_type,
                k=args.k,
                fetch_k=args.fetch_k,
                score_threshold=args.score_threshold,
                reranker_model=args.reranker_model,
                top_n=args.top_n,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        )
    else:
        answer_relevancy_average, faithfulness_average = rag_evaluate(
            backend_url=args.backend_url,
            llm=llm,
            ground_truth_file=args.ground_truth_file,
            use_openai_key=args.use_openai_key,
            embedding_model="BAAI/bge-large-en-v1.5",
            search_type=args.search_type,
            k=args.k,
            fetch_k=args.fetch_k,
            score_threshold=args.score_threshold,
            reranker_model=args.reranker_model,
            top_n=args.top_n,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    if os.path.exists("result_answer.jsonl"):
        os.remove("result_answer.jsonl")
