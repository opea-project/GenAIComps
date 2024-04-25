# GenAIEval
Evaluation, benchmark, and scorecard, targeting for performance on throughput and latency, accuracy on popular evaluation harness, safety, and hallucination

## Evaluation
### lm-evaluation-harness
We follow the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) and provide the command line usage and function call usage.
#### command line usage
```shell
python main.py --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cpu \
    --batch_size 8
```
#### function call usage
```python
from GenAIEval.evaluation.lm_evaluate import evaluate, LMEvalParser
args = LMevalParser(model = "hf", 
                    user_model = user_model,
                    tokenizer = tokenizer,
                    tasks = "hellaswag",
                    device = "cpu",
                    batch_size = 8,
                    )
results = evaluate(args)
```