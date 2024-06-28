# AutoRAG to evaluate the RAG system performance

AutoRAG is help to end-to-end evaluate the performance of the whole system. Currently, we support to evaluate the performance from 4 perspectives, answer_relevancy, faithfulness, context_recall, context_precision. Before using this service, the use should firstly prepare the groundtruth dataset in the [standard format](https://github.com/opea-project/GenAIEval/blob/main/evals/benchmark/ground_truth.jsonl). We also provide a [script](https://github.com/opea-project/GenAIEval/blob/main/evals/evaluation/autorag/data_generation/gen_eval_dataset.py) to automatically generate the groundtruth query and answer.

## Service preparation
The evaluation for the RAG system is based on the set up of the RAG services. Please follow [the steps](https://github.com/opea-project/GenAIExamples/blob/main/ChatQnA/README.md) to set up your RAG services.

## RAG evaluation
At this moment, we provide a solution that test the single group of parameters and multiple groups of parameters. For evaluating the single group of parameters, please firectly use [this script](https://github.com/opea-project/GenAIEval/blob/main/evals/evaluation/autorag/evaluation/ragas_evaluation_benchmark.py).

```bash
python -u ragas_evaluation_benchmark.py --ground_truth_file ground_truth.jsonl --search_type mmr --k 1 --fetch_k 5 --score_threshold 0.3 --top_n 1 --temperature 0.01 --top_k 5 --top_p 0.95 --repetition_penalty 1.1 --use_openai_key True
```

For evaluating multiple groups of parameters, please use [this script](https://github.com/opea-project/GenAIEval/blob/main/evals/benchmark/run_rag_benchmark.py). 
```bash
python -u run_rag_benchmark.py --config config.yaml
```

The group parameters should predefined in a `config.yaml`. It will pass available parameters to the RAG system.

## Notes
Due to some dependences issues, we can use OpenAI-series models to evaluate the RAG system from the four perspectives, answer_relevancy, faithfulness, context_recall, context_precision. If you want to use your local model to evaluate the RAG system, it can only support the evaluation for answer_relevancy and faithfulness.
