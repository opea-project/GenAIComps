#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#

from typing import Dict, Optional, Union

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_huggingface import HuggingFaceEndpoint


def format_ragas_metric_name(name: str):
    return f"{name} (ragas)"


class RAGASContextualPrecisionMetric:
    """This metric checks the contextual precision using Ragas."""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseLanguageModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = model

    def measure(self, test_case: Dict):
        try:
            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import context_precision

        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install ragas to use this metric. `pip install ragas`.")

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        if isinstance(self.model, str):
            chat_model = HuggingFaceEndpoint(
                endpoint_url=self.model,
                timeout=600,
            )
        else:
            chat_model = self.model
        chat_model = LangchainLLMWrapper(chat_model)
        # Create a dataset from the test case
        data = {
            "contexts": [test_case["retrieval_context"]],
            "question": [test_case["input"]],
            "ground_truth": [test_case["expected_output"]],
        }
        dataset = Dataset.from_dict(data)

        # Evaluate the dataset using Ragas
        scores = evaluate(dataset, metrics=[context_precision], llm=chat_model)
        # Ragas only does dataset-level comparisons
        context_precision_score = scores["context_precision"]
        self.success = context_precision_score >= self.threshold
        self.score = context_precision_score
        return self.score

    async def a_measure(self, test_case: Dict):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Contextual Precision")


class RAGASContextualRelevancyMetric:
    """This metric checks the contextual relevancy using Ragas."""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseLanguageModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = model

    async def a_measure(self, test_case: Dict):
        return self.measure(test_case)

    def measure(self, test_case: Dict):
        # sends to server
        try:
            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import context_relevancy

        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install ragas to use this metric. `pip install ragas`.")

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        if isinstance(self.model, str):
            chat_model = HuggingFaceEndpoint(
                endpoint_url=self.model,
                timeout=600,
            )
        else:
            chat_model = self.model
        chat_model = LangchainLLMWrapper(chat_model)
        # Create a dataset from the test case
        data = {
            "contexts": [test_case["retrieval_context"]],
            "question": [test_case["input"]],
        }
        dataset = Dataset.from_dict(data)

        # Evaluate the dataset using Ragas
        scores = evaluate(dataset, metrics=[context_relevancy], llm=chat_model)

        # Ragas only does dataset-level comparisons
        context_relevancy_score = scores["context_relevancy"]
        self.success = context_relevancy_score >= self.threshold
        self.score = context_relevancy_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Contextual Relevancy")


class RAGASAnswerRelevancyMetric:
    """This metric checks the answer relevancy using Ragas."""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseLanguageModel]] = "gpt-3.5-turbo",
        embeddings: Optional[Embeddings] = None,
    ):

        self.threshold = threshold
        self.model = model
        self.embeddings = embeddings

    async def a_measure(self, test_case: Dict):
        return self.measure(test_case)

    def measure(self, test_case: Dict):
        # sends to server
        try:
            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import answer_relevancy

        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install ragas to use this metric. `pip install ragas`.")

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        if isinstance(self.model, str):
            chat_model = HuggingFaceEndpoint(
                endpoint_url=self.model,
                timeout=600,
            )
        else:
            chat_model = self.model
        chat_model = LangchainLLMWrapper(chat_model)
        data = {
            "question": [test_case["input"]],
            "answer": [test_case["actual_output"]],
            "contexts": [test_case["retrieval_context"]],
        }
        dataset = Dataset.from_dict(data)

        scores = evaluate(
            dataset,
            metrics=[answer_relevancy],
            llm=chat_model,
            embeddings=self.embeddings,
        )
        answer_relevancy_score = scores["answer_relevancy"]
        self.success = answer_relevancy_score >= self.threshold
        self.score = answer_relevancy_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Answer Relevancy")


class RAGASFaithfulnessMetric:
    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseLanguageModel]] = "gpt-3.5-turbo",
    ):

        self.threshold = threshold
        self.model = model

    async def a_measure(self, test_case: Dict):
        return self.measure(test_case)

    def measure(self, test_case: Dict):
        # sends to server
        try:
            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import faithfulness

        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install ragas to use this metric. `pip install ragas`.")

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        if isinstance(self.model, str):
            chat_model = HuggingFaceEndpoint(
                endpoint_url=self.model,
                timeout=600,
            )
        else:
            chat_model = self.model
        chat_model = LangchainLLMWrapper(chat_model)
        data = {
            "contexts": [test_case["retrieval_context"]],
            "question": [test_case["input"]],
            "answer": [test_case["actual_output"]],
        }
        dataset = Dataset.from_dict(data)

        scores = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=chat_model,
        )
        faithfulness_score = scores["faithfulness"]
        self.success = faithfulness_score >= self.threshold
        self.score = faithfulness_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Faithfulness")


class RAGASContextualRecallMetric:
    """This metric checks the context recall using Ragas."""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseLanguageModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = model

    async def a_measure(self, test_case: Dict):
        return self.measure(test_case)

    def measure(self, test_case: Dict):
        # sends to server
        try:
            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import context_recall

        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install ragas to use this metric. `pip install ragas`.")

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        if isinstance(self.model, str):
            chat_model = HuggingFaceEndpoint(
                endpoint_url=self.model,
                timeout=600,
            )
        else:
            chat_model = self.model
        chat_model = LangchainLLMWrapper(chat_model)
        data = {
            "question": [test_case["input"]],
            "ground_truth": [test_case["expected_output"]],
            "contexts": [test_case["retrieval_context"]],
        }
        dataset = Dataset.from_dict(data)

        scores = evaluate(
            dataset,
            [context_recall],
            llm=chat_model,
        )
        context_recall_score = scores["context_recall"]
        self.success = context_recall_score >= self.threshold
        self.score = context_recall_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Contextual Recall")


class RagasMetric:
    """This metric checks if the output is more than 3 letters."""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseLanguageModel]] = "gpt-3.5-turbo",
        embeddings: Optional[Embeddings] = None,
    ):

        self.threshold = threshold
        self.model = model
        self.embeddings = embeddings

    async def a_measure(self, test_case: Dict):
        return self.measure(test_case)

    def measure(self, test_case: Dict):
        # sends to server
        try:
            from ragas import evaluate
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install ragas to use this metric. `pip install ragas`.")

        try:
            # How do i make sure this isn't just huggingface dataset
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Create a dataset from the test case
        # Convert the Dict to a format compatible with Dataset
        score_breakdown = {}
        metrics = [
            RAGASContextualPrecisionMetric(model=self.model),
            RAGASContextualRecallMetric(model=self.model),
            RAGASFaithfulnessMetric(model=self.model),
            RAGASAnswerRelevancyMetric(model=self.model, embeddings=self.embeddings),
        ]

        for metric in metrics:
            score = metric.measure(test_case)
            score_breakdown[metric.__name__] = score

        ragas_score = sum(score_breakdown.values()) / len(score_breakdown)

        self.success = ragas_score >= self.threshold
        self.score = ragas_score
        self.score_breakdown = score_breakdown
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "RAGAS"
