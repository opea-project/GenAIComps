import json
import os
import pandas as pd
import argparse
from langchain_core.prompts import PromptTemplate
from model import setup_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--trace_json', type=str, required=True, help='Path to trace json file')
    parser.add_argument('--ref_answers', type=str, required=True, help='Path to reference answers file')
    parser.add_argument('--model', type=str, required=True, help='Path to sentence transformer model')
    parser.add_argument('--output', type=str, required=True, help='Path to output file')
    parser.add_argument('--threshold', type=float, default=0.7, help='Threshold for similarity score')
    parser.add_argument("--hf_tgi", action="store_true")
    parser.add_argument("--hf_pipe", action="store_true")
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--use_llm_evaluator", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    return args

def load_trace(trace_json):
    data = []
    with open(trace_json, 'r') as f:
        data = json.load(f)
    return data

def response_contain_ref_answer(response, ref_answer):
    if ref_answer in response:
        return True
    else:
        return False

def common_words_ratio(response, ref_answer):
    response = response.lower()
    ref_answer = ref_answer.lower()
    response = response.split()
    ref_answer = ref_answer.split()
    common = set(response) & set(ref_answer)
    score = len(common) / len(ref_answer)
    return score

def similarity_score(response, ref_answer, model):
    response = response.lower()
    ref_answer = ref_answer.lower()
    embeddings1 = model.encode(response, convert_to_tensor=True)
    embeddings2 = model.encode(ref_answer, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_score = st_util.cos_sim(embeddings1, embeddings2)    
    return cosine_score

EVALUATOR_TEMPLATE = """
Given the question and the reference answer, determine if the response answers the question correctly or not.
If the response is a number, then it is correct when it is very close to the reference answer.
You must output YES or NO.
Question: {query}
Response: {response}
Reference Answer: {ref_answer}
"""


def llm_evaluator(query, response, ref_answer, model, template, args):
    prompt = PromptTemplate.from_template(template)
    evaluator = prompt | model
    output = evaluator.invoke({
        "query": query,
        "response": response, 
        "ref_answer": ref_answer})
    
    if args.llm_engine == 'openai':
        output = output.content
    print(output)

    if output == 'YES':
        return 'correct'
    else:
        return 'incorrect'


def count_iterations(data, eval_result):
    num_iter = []
    for trace, result in zip(data, eval_result):
        if result == "correct":
            iteration = 0
            # count num of "replan" steps
            for step in trace['trace']:
                if 'replan' in step:
                    iteration += 1
            num_iter.append(iteration)
    
    print("average # of iterations for successful completion: {}".\
    format(int(sum(num_iter)/len(num_iter))))



def classic_evaluator(response, ref_answer, model, threshold):
    if len(response.split(' ')) == 1:
        # single word response, exact match
        if response.lower() == ref_answer.lower():
            return 'correct'
        else:
            return 'incorrect'
    else:
        # multi-word response, check for similarity
        if response_contain_ref_answer(response, ref_answer):
            return 'correct'
        elif similarity_score(response, ref_answer, model) > threshold:
            return 'correct'
        else:
            return 'incorrect'

def evaluate_trace(data, ref_answers, model, args):
    no_response = 0
    correct = 0
    incorrect = 0
    eval_result = []
    for trace, ref_answer in zip(data, ref_answers):
        query = trace['query']
        if "Recursion limit" in trace['trace']:
            no_response += 1 
            eval_result.append('no_response')   
        else:
            last_step = trace['trace'][-1] # should be output from replanner
            response = last_step['replan']['response']
            print('Query:', query)
            print('Response:', response)
            print('Reference Answer:', ref_answer)
            if args.use_llm_evaluator:
                result = llm_evaluator(query, response, ref_answer, model, EVALUATOR_TEMPLATE, args)
            else:
                result = classic_evaluator(response, ref_answer, model, args.threshold)

            eval_result.append(result)
            if result == 'correct':
                correct += 1
            else:
                incorrect += 1

    print(f'Correct: {correct}, Incorrect: {incorrect}, No Response: {no_response}')
    return eval_result

def main():
    args = get_args()
    data = load_trace(os.path.join(args.data_dir,args.trace_json))
    df = pd.read_csv(os.path.join(args.data_dir,args.ref_answers))
    ref_answers = df['answer'].to_list()

    if args.use_llm_evaluator:
        model = setup_model(args)
    else:
        from sentence_transformers import SentenceTransformer 
        from sentence_transformers import util as st_util
        model = SentenceTransformer(args.model)

    eval_result = evaluate_trace(data, ref_answers, model, args)
    df['eval_result'] = eval_result

    df.to_csv(os.path.join(args.data_dir, args.output), index=False)

    # eval_result = df['eval_result'].to_list()
    # print(eval_result)
    count_iterations(data, eval_result)

if __name__ == '__main__':
    main()