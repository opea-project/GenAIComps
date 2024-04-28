from GenAIEval.evaluation.lm_evaluation_harness import evaluate, setup_parser

# from GenAIEval.evaluation.bigcode_evaluation_harness import evaluate, setup_parser


def main():
    eval_args = setup_parser()
    results = evaluate(eval_args)


if __name__ == "__main__":
    main()
