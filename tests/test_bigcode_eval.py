#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

from evals.evaluation.bigcode_evaluation_harness import BigcodeEvalParser, evaluate


class TestLMEval(unittest.TestCase):
    def test_lm_eval(self):
        model_name_or_path = "bigcode/tiny_starcoder_py"
        user_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, truncation_side="left", padding_side="right")
        args = BigcodeEvalParser(
            user_model=user_model,
            tokenizer=tokenizer,
            tasks="humaneval",
            n_samples=2,
            batch_size=2,
            allow_code_execution=True,
            limit=10,
        )
        results = evaluate(args)
        self.assertEqual(results["humaneval"]["pass@1"], 0.1)


if __name__ == "__main__":
    unittest.main()
