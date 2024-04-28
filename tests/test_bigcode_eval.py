import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

from GenAIEval.evaluation.bigcode_evaluation_harness import (BigcodeEvalParser,
                                                             evaluate)


class TestLMEval(unittest.TestCase):
    def test_lm_eval(self):
        model_name_or_path = "codeparrot/codeparrot-small"
        user_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, truncation_side="left", padding_side="right"
        )
        args = BigcodeEvalParser(
            user_model=user_model,
            tokenizer=tokenizer,
            tasks="humaneval",
            n_samples=20,
            batch_size=10,
            allow_code_execution=True,
            limit=20,
        )
        results = evaluate(args)
        self.assertEqual(results["humaneval"]["pass@1"], 0.05)


if __name__ == "__main__":
    unittest.main()
