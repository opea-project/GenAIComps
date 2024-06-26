#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
import os
from typing import Dict, Optional, Union

from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel


def format_ragas_metric_name(name: str):
    return f"{name} (ragas)"


class RagasMetric:
    """This metric checks if the output is more than 3 letters."""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseLanguageModel]] = None,
        embeddings: Optional[Embeddings] = None,
        metrics: Optional[list[str]] = None,
    ):

        self.threshold = threshold
        self.model = model
        self.embeddings = embeddings
        self.metrics = metrics
        self.validated_list = ["answer_relevancy", "faithfulness"]

    async def a_measure(self, test_case: Dict):
        return self.measure(test_case)

    def measure(self, test_case: Dict):

        # sends to server
        try:
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, faithfulness

        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install ragas to use this metric. `pip install ragas`.")

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")
        self.metrics_instance = {
            "answer_relevancy": answer_relevancy,
            "faithfulness": faithfulness,
        }

        # Set LLM model
        openai_key = os.getenv("OPENAI_API_KEY", None)
        if openai_key is not None:
            print("OPENAI_API_KEY is provided, ragas initializes the model by OpenAI.")
            self.model = None
        if isinstance(self.model, str):
            chat_model = HuggingFaceEndpoint(
                endpoint_url=self.model,
                timeout=600,
            )
        else:
            chat_model = self.model
        # Create a dataset from the test case
        # Convert the Dict to a format compatible with Dataset
        if self.metrics is not None:
            tmp_metrics = []
            # check supported list
            for metric in self.metrics:
                if metric not in self.validated_list:
                    raise ValueError(
                        "metric should be in supported list {}. ".format(self.validated_list)
                        + "ClientResponseError raised with LangchainLLM "
                        + "when context_precision, context_recall ran. "
                        + "Here are the related issues described in ragas "
                        "https://github.com/explodinggradients/ragas/issues/934, "
                        + "https://github.com/explodinggradients/ragas/issues/664."
                    )
                else:
                    if metric == "answer_relevancy" and self.embeddings is None:
                        raise ValueError("answer_relevancy metric need provide embeddings model.")
                    tmp_metrics.append(metric)
            self.metrics = tmp_metrics
        else:
            self.metrics = [
                answer_relevancy,
                faithfulness,
            ]

        data = {
            "question": [test_case["input"]],
            "contexts": [test_case["retrieval_context"]],
            "answer": [test_case["actual_output"]],
            "ground_truth": [test_case["expected_output"]],
        }
        dataset = Dataset.from_dict(data)

        self.score = evaluate(
            dataset,
            metrics=self.metrics,
            llm=chat_model,
            embeddings=self.embeddings,
        )
        print(self.score)
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "RAGAS"
