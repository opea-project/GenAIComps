# GenAIEval
Evaluation, benchmark, and scorecard, targeting for performance on throughput and latency, accuracy on popular evaluation harness, safety, and hallucination

## Installation
```shell
git clone https://github.com/opea-project/GenAIEval
cd GenAIEval
pip install -e .
```
## Evaluation
### lm-evaluation-harness
For evaluating the models on text-generation tasks, we follow the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/) and provide the command line usage and function call usage. Over 60 standard academic benchmarks for LLMs, with hundreds of [subtasks and variants](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.2/lm_eval/tasks) implemented, such as `ARC`, `HellaSwag`, `MMLU`, `TruthfulQA`, `Winogrande`, `GSM8K` and so on.
#### command line usage

##### Gaudi2
```shell
python main.py \
    --model gaudi-hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device hpu \
    --batch_size 8
```


##### CPU
```shell
python main.py \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cpu \
    --batch_size 8
```
#### function call usage
```python
from GenAIEval.evaluation.lm_evaluation_harness import evaluate, LMEvalParser
args = LMevalParser(model = "hf", 
                    user_model = user_model,
                    tokenizer = tokenizer,
                    tasks = "hellaswag",
                    device = "cpu",
                    batch_size = 8,
                    )
results = evaluate(args)
```

### bigcode-evaluation-harness
For evaluating the models on coding tasks or specifically coding LLMs, we follow the [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) and provide the command line usage and function call usage. [HumanEval](https://huggingface.co/datasets/openai_humaneval), [HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus), [InstructHumanEval](https://huggingface.co/datasets/codeparrot/instructhumaneval), [APPS](https://huggingface.co/datasets/codeparrot/apps), [MBPP](https://huggingface.co/datasets/mbpp), [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus), and [DS-1000](https://github.com/HKUNLP/DS-1000/) for both completion (left-to-right) and insertion (FIM) mode are available.
#### command line usage
There is a small code change in `main.py` regarding the import path.
```diff
- from GenAIEval.evaluation.lm_evaluation_harness import evaluate, setup_parser
+ from GenAIEval.evaluation.bigcode_evaluation_harness import evaluate, setup_parser
```
```shell
python main.py \
    --model "codeparrot/codeparrot-small" \
    --tasks "humaneval" \
    --n_samples 100 \
    --batch_size 10 \
    --allow_code_execution \
```
#### function call usage
```python
from GenAIEval.evaluation.bigcode_evaluation_harness import evaluate, BigcodeEvalParser
args = BigcodeEvalParser(
                    user_model = user_model,
                    tokenizer = tokenizer,
                    tasks = "humaneval",
                    n_samples = 100,
                    batch_size = 10,
                    allow_code_execution=True,
                    )
results = evaluate(args)
```
