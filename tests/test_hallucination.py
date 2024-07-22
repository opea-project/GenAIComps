#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from evals.metrics.hallucination import HallucinationMetric


class TestHallucinationMetric(unittest.TestCase):

    @unittest.skip("need pass localhost id")
    def test_hallucination(self):
        # Replace this with the actual output from your LLM application
        actual_output = "A blond drinking water in public."

        # Replace this with the actual documents that you are passing as input to your LLM.
        context = ["A man with blond-hair, and a brown shirt drinking out of a public water fountain."]

        metric = HallucinationMetric(threshold=0.5, model="http://localhost:8008/generate")
        test_case = {"input": "What was the blond doing?", "actual_output": actual_output, "context": context}

        metric.measure(test_case)
        print(metric.score)


if __name__ == "__main__":
    unittest.main()
