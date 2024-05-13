#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

from GenAIEval.evaluation.bigcode_evaluation_harness import BigcodeEvalParser, evaluate


class TestLMEval(unittest.TestCase):
    def test_lm_eval(self):
        model_name_or_path = "codeparrot/codeparrot-small"
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
