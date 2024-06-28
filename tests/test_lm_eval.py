#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

from evals.evaluation.lm_evaluation_harness import LMEvalParser, evaluate


class TestLMEval(unittest.TestCase):
    def test_lm_eval(self):
        model_name_or_path = "facebook/opt-125m"
        user_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        args = LMEvalParser(
            model="hf",
            user_model=user_model,
            tokenizer=tokenizer,
            tasks="lambada_openai",
            device="cpu",
            batch_size=1,
            limit=5,
            trust_remote_code=True,
        )
        results = evaluate(args)
        self.assertEqual(results["results"]["lambada_openai"]["acc,none"], 0.6)


if __name__ == "__main__":
    unittest.main()
