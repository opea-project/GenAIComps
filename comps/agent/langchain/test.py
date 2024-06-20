import argparse
import os
from src.utils import format_date, get_args
import pandas as pd
import json
import requests
import traceback


def test_agent_local(args):
    from src.agent import instantiate_agent

    df = pd.DataFrame({
    "query": ["what was the total trading volume of metlife?", "What is the weather today in Austin?"]
    })
    #df = pd.read_csv(os.path.join(args.filedir, args.filename))
    # df = df.sample(n=2, random_state=42)

    agent = instantiate_agent(args, strategy=args.strategy)
    app = agent.app

    config = {"recursion_limit": args.recursion_limit}

    traces = []
    for _, row in df.iterrows():
        print('Query: ', row['query'])
        initial_state = {
            "input": row['query'],
            "plan_errors":[],
            "past_steps":[]
        }
        try:
            trace = {"query": row['query'], "trace":[]}
            for event in app.stream(initial_state, config=config):
                trace["trace"].append(event)
                for k, v in event.items():
                    print("{}: {}".format(k,v))
                
            traces.append(trace)
        except Exception as e:
            print(str(e), str(traceback.format_exc()))
            traces.append({"query": row['query'], "trace":str(e)})

        print('-'*50)
    
    df['trace'] = traces
    df.to_csv(os.path.join(args.filedir, args.output), index=False)

def test_agent_http(args):
    proxies = {"http": ""}
    ip_addr = args.ip_addr
    url = f"http://{ip_addr}:9000/v1/chat/completions"
    
    def process_request(query):
        content = json.dumps({"query": query})
        print(content)
        try:
            resp = requests.post(url=url, data=content, proxies=proxies)
            ret = resp.text
            resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
        except requests.exceptions.RequestException as e:
            ret = f"An error occurred:{e}"
        print(ret)
        return ret
        
    #df = pd.read_csv(os.path.join(args.filedir, args.filename))
    df = pd.DataFrame({
    "query": ["what was the total trading volume of metlife during the first week of february?"]
    })
    traces = []
    for _, row in df.iterrows():        
        ret = process_request(row['query'])
        trace = {"query": row['query'], "trace":ret}
        traces.append(trace)

    df['trace'] = traces
    df.to_csv(os.path.join(args.filedir, args.output), index=False)
        
def test_llm(args):
    from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
    
    generation_params = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "top_k": 50,
            # "top_p":0.8,
            "temperature": 1.0,
            "repetition_penalty": 1.03,
            "return_full_text":False,
            # "eos_token_id":model.config.eos_token_id,
        }
    
    llm = HuggingFaceEndpoint(
        endpoint_url=args.llm_endpoint_url, ## endpoint_url = "localhost:8080"
        task="text-generation",
        **generation_params
    )
    query = '''
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Given the OBJECTIVE, create a step-by-step plan by only using the tools listed below. Each step must contain 3 elements: tool, input_schema and output_key. Each step can use ONLY ONE tool. Do NOT combine tool calls in one step. A step in the plan can receive the output from a previous step as input. Use <> to enclose an input when using an output from a previous step. Read tool descriptions carefully and only come up with steps that the tools can do. 

Tools: 
1. def search(query: str) -> str:
    """
    Search the web for financial information.

    Args:
    query (str): The search query. Must be a simple string, can NOT be f-string or string added together.

    Returns:
    str: text paragraphs relevant to the query, does not return exact answers.
    """

2. def profit_calculator(principle: float, sold: float)->float:
    """
    Calculates profit percentage from principle and sold values.
    
    Args:
    principle (float): the money invested in the beginning.
    sold (float): the money earned after selling the investment.

    Returns:
    float: the profit percentage.
    """

3. def interest_calculator(annual_rate:float, principle:float):
    """
    Calculates the interest for a given annual rate and principle.

    Args:
    annual_rate (float): the annual interest rate.
    principle (float): the principle amount.

    Returns:
    float: the interest amount.
    """

4. def ticker_lookup(entity: str) -> str:
    """
    This function returns the ticker symbol for a stock, ETF, or an index.

    Args:
    entity (str): The name of the stock, ETF, or index.

    Returns:
    str: the ticker symbol.
    """

5. def get_trade_info_for_single_ticker(ticker: str, from_date: str, span:str, info_to_seek:str)->str:
    """
    This function returns a single piece of information from the trading aggregates for a SINGLE stock or index ticker.    The information can be open_price, close_price, low_price, high_price, average_price, trading_volume, num_transaction.
    No other info can be obtained with this tool.

    Args:
    ticker (str): the stock or index ticker symbol. ONLY one ticker is allowed.
    from_date (str): the start date for the aggregates, must be in the YYYY-MM-DD format.
    span (str): the time span for the aggregates, can be "day", "week", "month", "quarter" or "year.
    info_to_seek (str): can only be one of the following: open_price, close_price,     low_price, high_price, average_price, trading_volume, num_transaction.

    Returns:
    str: the single piece of trade info for the ticker
    """


You MUST strictly follow the output format below. You MUST NOT include any other text!
{  
  "steps": [
    {"tool": str, "input_schema": {input_schema}, "output_key": str}
  ]
}

Do NOT use f-strings or string operations in the steps. Do NOT include your thought or explanations in your output! Do NOT add comments to the steps!

Objective can be time sensitive. Pay attention to today's date when composing the plan. Today's date is: ['2022-09-01'] 

OBJECTIVE: How do you calculate the WACC?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
</|end_text|>"
'''
    print(llm(query))


if __name__ == "__main__":
    args1, _ = get_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_test", action="store_true", help="Test with local mode")
    parser.add_argument("--endpoint_test", action="store_true", help="Test with endpoint mode")
    parser.add_argument("--llm_test", action="store_true", help="Test with llm")
    parser.add_argument("--filedir", type=str, default="./", help="test file directory")
    parser.add_argument("--filename", type=str, default="query.csv", help="query_list_file")
    parser.add_argument("--output", type=str, default="output.csv", help="query_list_file")
    parser.add_argument("--strategy", type=str, default="react", choices=["react", "planexec"])

    args, _ = parser.parse_known_args()
    
    for key, value in vars(args1).items():
        setattr(args, key, value)
    
    args.ip_addr = "100.83.111.250"
    if args.local_test:
        test_agent_local(args)
    elif args.endpoint_test:
        test_agent_http(args)
    elif args.llm_test:
        test_llm(args)
    else:
        print("Please specify the test type")