from transformers import AutoModelForCausalLM, AutoTokenizer
model_name_or_path = "facebook/opt-125m"
user_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

from GenAIEval.evaluation.lm_evaluation_harness import evaluate, setup_parser, parse_eval_args
def parse_args():
    parser = setup_parser()
    args = parse_eval_args(parser)
    return args

def main():
    args = parse_args()
    results = evaluate(args)

if __name__ == "__main__":
    main()