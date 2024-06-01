# GenAIEval
Evaluation, benchmark, and scorecard, targeting for performance on throughput and latency, accuracy on popular evaluation harness, safety, and hallucination

## Installation

- Install from Pypi

```bash
pip install -r requirements.txt
pip install opea-eval
```
> notes: We have to install requirements.txt at first, cause Pypi can't have direct dependency with specific commit. 

- Build from Source

```bash
git clone https://github.com/opea-project/GenAIEval
cd GenAIEval
pip install -r requirements.txt
pip install -e .
```

## Evaluation
### lm-evaluation-harness
For evaluating the models on text-generation tasks, we follow the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/) and provide the command line usage and function call usage. Over 60 standard academic benchmarks for LLMs, with hundreds of [subtasks and variants](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.2/lm_eval/tasks) implemented, such as `ARC`, `HellaSwag`, `MMLU`, `TruthfulQA`, `Winogrande`, `GSM8K` and so on.
#### command line usage

##### Gaudi2
```shell

# pip install --upgrade-strategy eager optimum[habana]
cd evals/evaluation/lm_evaluation_harness/examples
python main.py \
    --model gaudi-hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device hpu \
    --batch_size 8
```


##### CPU
```shell

cd evals/evaluation/lm_evaluation_harness/examples
python main.py \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cpu \
    --batch_size 8
```
#### function call usage
```python
from evals.evaluation.lm_evaluation_harness import LMEvalParser, evaluate

args = LMevalParser(
    model="hf",
    user_model=user_model,
    tokenizer=tokenizer,
    tasks="hellaswag",
    device="cpu",
    batch_size=8,
)
results = evaluate(args)
```

#### remote service usage

1. setup a separate server with [GenAIComps](https://github.com/opea-project/GenAIComps/tree/main/comps/llms/lm-eval)

```
# build cpu docker
docker build -f Dockerfile.cpu -t opea/lm-eval:latest .

# start the server
docker run -p 9006:9006 --ipc=host  -e MODEL="hf" -e MODEL_ARGS="pretrained=Intel/neural-chat-7b-v3-3" -e DEVICE="cpu" opea/lm-eval:latest
```

2. evaluate the model

- set `base_url`, `tokenizer` and `--model genai-hf`

```
cd evals/evaluation/lm_evaluation_harness/examples

python main.py \
    --model genai-hf \
    --model_args "base_url=http://{your_ip}:9006,tokenizer=Intel/neural-chat-7b-v3-3" \
    --tasks  "lambada_openai" \
    --batch_size 2
```

### bigcode-evaluation-harness
For evaluating the models on coding tasks or specifically coding LLMs, we follow the [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) and provide the command line usage and function call usage. [HumanEval](https://huggingface.co/datasets/openai_humaneval), [HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus), [InstructHumanEval](https://huggingface.co/datasets/codeparrot/instructhumaneval), [APPS](https://huggingface.co/datasets/codeparrot/apps), [MBPP](https://huggingface.co/datasets/mbpp), [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus), and [DS-1000](https://github.com/HKUNLP/DS-1000/) for both completion (left-to-right) and insertion (FIM) mode are available.
#### command line usage

```shell
cd evals/evaluation/bigcode_evaluation_harness/examples
python main.py \
    --model "codeparrot/codeparrot-small" \
    --tasks "humaneval" \
    --n_samples 100 \
    --batch_size 10 \
    --allow_code_execution
```
#### function call usage
```python
from evals.evaluation.bigcode_evaluation_harness import BigcodeEvalParser, evaluate

args = BigcodeEvalParser(
    user_model=user_model,
    tokenizer=tokenizer,
    tasks="humaneval",
    n_samples=100,
    batch_size=10,
    allow_code_execution=True,
)
results = evaluate(args)
```
