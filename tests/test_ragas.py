#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
import unittest

from evals.metrics.ragas import RagasMetric


class TestRagasMetric(unittest.TestCase):

    @unittest.skip("need assign localhost id")
    def test_ragas(self):
        # Replace this with the actual output from your LLM application
        actual_output = "We offer a 30-day full refund at no extra cost."

        # Replace this with the expected output from your RAG generator
        expected_output = "You are eligible for a 30 day full refund at no extra cost."

        # Replace this with the actual retrieved context from your RAG pipeline
        retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        metric = RagasMetric(threshold=0.5, model="http://localhost:8008", embeddings=embeddings)
        test_case = {
            "input": "What if these shoes don't fit?",
            "actual_output": actual_output,
            "expected_output": expected_output,
            "retrieval_context": retrieval_context,
        }

        metric.measure(test_case)
        print(metric.score)


if __name__ == "__main__":
    unittest.main()
