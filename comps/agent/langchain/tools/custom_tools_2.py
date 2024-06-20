# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from typing import List

SEARCH_TOOL_DESCRIPTION = '''\
def search(query: str) -> str:
    """
    Search the web for financial information.

    Args:
    query (str): The search query. Must be a simple string, can NOT be f-string or string added together.

    Returns:
    str: text paragraphs relevant to the query, does not return exact answers.
    """
'''


def search(query: str) -> str:
    # from langchain_community.tools.tavily_search import TavilySearchResults
    # tool = TavilySearchResults(max_results=3)
    # res = tool.invoke({"query": query})
    import os

    from tavily import TavilyClient

    TAVILYKEY = os.getenv("TAVILY_API_KEY")
    tavily = TavilyClient(api_key=TAVILYKEY)
    search_params = {"search_depth": "advanced", "max_results": 3, "include_answer": True}

    ret_text = ""

    try:
        print("Query:\n", query)
        res = tavily.search(query=query, **search_params)
        answer = res["answer"]
        print("Answer:\n", answer)

        # for i, r in enumerate(res['results']):
        #     print('Content #{}:\n{}'.format(i,r['content']))

        query = answer
        ret_text = ret_text + answer + "\n"
        # print('-'*50)
    except Exception as e:
        ret_text = "Exception occurred during search: {}".format(str(e))
        print(str(e))

    return ret_text


def search_google(query: str) -> str:
    from langchain_community.utilities import GoogleSearchAPIWrapper
    from langchain_core.tools import Tool

    search = GoogleSearchAPIWrapper(k=10)

    tool = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=search.run,
    )
    res = tool.run(query)
    return res


def search_ddg(query: str) -> str:
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

    wrapper = DuckDuckGoSearchAPIWrapper(region="us-en", time="d", max_results=3)
    # time can be d, y
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")  # source can be text or news
    results = search.run(query)

    ret_text = ""

    for res in results:
        ret_text = ret_text + res + "\n"
    return ret_text


PROFIT_CALCULATOR_DESCRIPTION = '''\
def profit_calculator(principle: float, sold: float)->float:
    """
    Calculates profit percentage from principle and sold values.

    Args:
    principle (float): the money invested in the beginning.
    sold (float): the money earned after selling the investment.

    Returns:
    float: the profit percentage.
    """
'''


def profit_calculator(principle: float, sold: float) -> float:
    return "{:.2f}".format((sold - principle) / principle * 100)


INTEREST_CALCULATOR_DESCRIPTION = '''\
def interest_calculator(annual_rate:float, principle:float):
    """
    Calculates the interest for a given annual rate and principle.

    Args:
    annual_rate (float): the annual interest rate.
    principle (float): the principle amount.

    Returns:
    float: the interest amount.
    """
'''


def interest_calculator(annual_rate: float, principle: float):
    return "{:.2f}".format(principle * annual_rate / 12)


TICKER_AGGREGATES_DESCRIPTION = '''\
def get_aggregates_for_ticker(ticker: str, from_date:str, span:str):
    """
    This function uses the Polygon.io API to get trading aggregates for a stock or index ticker.
    Args:
    ticker (str): the stock or index ticker symbol. ONLY one ticker is allowed.
    from_date (str): the start date for the aggregates, must be in the YYYY-MM-DD format.
    to_date (str): the end date for the aggregates, must be in the YYYY-MM-DD format.
    span (str): the time span for the aggregates, can be "day", "week", "month", "quarter" or "year.

    Returns:
    str: a json string contains open, close, highest, lowest prices during the span, and the number of.

    Example usage:
    To get the trading aggregates for Apple Inc. (AAPL) from April 1, 2024 to April 5, 2024 with a weekly span:

    get_aggregates_for_ticker("APPL", "2024-04-01", "week")

    This tool call would return the json string below:
    {
        'open_price': 159.94,
        'close_price': 164.9,
        'low': 155.98,
        'high': 165,
        'num_trade': 2453521,
        'volume': 267648190,
        'average_price': 160.74
    }
    """
'''


def get_aggregates_for_ticker(ticker: str, from_date: str, span: str):
    import time

    from langchain_community.tools.polygon.aggregates import PolygonAggregates, PolygonAggregatesSchema
    from langchain_community.utilities.polygon import PolygonAPIWrapper

    api_wrapper = PolygonAPIWrapper()

    # Define param
    params = PolygonAggregatesSchema(
        ticker=ticker,
        timespan=span,
        timespan_multiplier=1,
        from_date=from_date,
        to_date=from_date,
    )
    # Get aggregates for ticker
    aggregates_tool = PolygonAggregates(api_wrapper=api_wrapper)
    aggregates = aggregates_tool.run(tool_input=params.dict())
    aggregates_json = json.loads(aggregates)

    volume = aggregates_json[0]["v"]
    avg_price = aggregates_json[0]["vw"]
    n_trade = aggregates_json[0]["n"]
    low = aggregates_json[0]["l"]
    open_price = aggregates_json[0]["o"]
    # market_cap = volume * avg_price

    ret_dict = {
        "open_price": open_price,
        "close_price": aggregates_json[0]["c"],
        "low_price": low,
        "high_price": aggregates_json[0]["h"],
        "num_transaction": n_trade,
        "trading_volume": volume,
        "average_price": avg_price,
        # 'market_cap': market_cap
    }

    ret_json = json.dumps(ret_dict)

    time.sleep(30)  # wait for 20 secs due to limitation of free Polygon API

    return ret_json


def extract_info_from_aggregates(aggregates: str, info_to_seek: str) -> str:
    try:
        aggregates_json = json.loads(aggregates)
        return aggregates_json[info_to_seek]
    except Exception as e:
        return str(e)


GET_TRADE_INFO_TOOL = '''\
def get_trade_info_for_single_ticker(ticker: str, from_date: str, span:str, info_to_seek:str)->str:
    """
    This function returns a single piece of information from the trading aggregates for a SINGLE stock or index ticker.\
    The information can be open_price, close_price, low_price, high_price, average_price, trading_volume, num_transaction.
    No other info can be obtained with this tool.

    Args:
    ticker (str): the stock or index ticker symbol. ONLY one ticker is allowed.
    from_date (str): the start date for the aggregates, must be in the YYYY-MM-DD format.
    span (str): the time span for the aggregates, can be "day", "week", "month", "quarter" or "year.
    info_to_seek (str): can only be one of the following: open_price, close_price, \
    low_price, high_price, average_price, trading_volume, num_transaction.

    Returns:
    str: the single piece of trade info for the ticker
    """
'''


def get_trade_info_for_single_ticker(ticker: str, from_date: str, span: str, info_to_seek: str) -> str:
    # first check input
    if "Ticker lookup tool cannot find ticker for" in ticker:
        return "Upstream ticker lookup failed, so cannot get aggregates."
    else:
        aggregates = get_aggregates_for_ticker(ticker, from_date, span)
        if aggregates:
            return extract_info_from_aggregates(aggregates, info_to_seek)
        else:
            return "Cannot find trade info for ticker {}".format(ticker)


INFO_EXTRACTION_TOOL = '''\
def extract_info_from_aggregates(aggregates:str, info_to_seek:str) -> str:
    """
    This function extracts ONE SINGLE piece of info from the json string returned by the get_aggregates_for_ticker function.

    Args:
    aggregates (str): the json string returned by the get_aggregates_for_ticker function.
    info_to_seek (str): the information to extract from the json string, for example, volume.

    Returns:
    str: the extracted information.
    """
 '''


TICKER_DICT = {
    "foghorn therapeutics": "FHTX",
}

TICKER_LOOKUP_TOOL_DESCRIPTION = '''\
def ticker_lookup(entity: str) -> str:
    """
    This function returns the ticker symbol for a stock, ETF, or an index.

    Args:
    entity (str): The name of the stock, ETF, or index.

    Returns:
    str: the ticker symbol.
    """
'''


def ticker_lookup(entity: str) -> str:
    import requests

    if "ETF" in entity.upper():
        entity = entity.upper().replace(" ETF", "")

    if entity.lower() in TICKER_DICT:
        print("use ticker_dict")
        return TICKER_DICT[entity.lower()]
    else:
        try:
            print("try ticker lookup api")
            # api-endpoint
            URL = "https://ticker-2e1ica8b9.now.sh/keyword/" + entity
            # sending get request and saving the response as response object
            r = requests.get(url=URL)

            # print(r)
            # extracting data in json format
            data = r.json()[0]["symbol"]

            print("ticker symbol for {} is: {}".format(entity, data))
            return data

        except:
            if len(entity) <= 4:
                print("entity seems to be ticker symbol")
                return entity.upper()
            else:
                return "Ticker lookup tool cannot find ticker for {}".format(entity)


def ticker_list_lookup(entity_list) -> dict:
    # entity_list should be a list of entities
    # or a str with entities separated by commas
    # or a str with entities separated by spaces
    import requests

    if type(entity_list) == str:
        if "," in entity_list:
            entity_list = entity_list.split(",")
        else:
            entity_list = entity_list.split(" ")

    tickers_dict = {}
    for entity in entity_list:
        print("Looking up ticker symbol for: ", entity)
        entity = entity.replace(" ", "")  # remove spaces
        tickers[entity] = ticker_lookup(entity)

    return tickers_dict


TIME_LOOKUP_TOOL_DESCRIPTION = '''\
def now(query:str) -> str:
    """
    This function returns the date and time when the user query was asked.

    Returns:
    str: The date and time when the query was issued.
    """
'''


def now(query: str) -> str:
    return query.split("@")[-1]


def tools_descriptions():
    tools = [
        # TIME_LOOKUP_TOOL_DESCRIPTION,
        SEARCH_TOOL_DESCRIPTION,
        PROFIT_CALCULATOR_DESCRIPTION,
        INTEREST_CALCULATOR_DESCRIPTION,
        TICKER_LOOKUP_TOOL_DESCRIPTION,
        # TICKER_AGGREGATES_DESCRIPTION,
        # INFO_EXTRACTION_TOOL,
        GET_TRADE_INFO_TOOL,
    ]
    return tools


def get_valid_tools_and_args():
    tools = tools_descriptions()
    valid_tools = []
    valid_args = {}
    for i, tool in enumerate(tools):
        valid_tools.append(tool.split("(")[0].split(" ")[1])
        args_list = tool.split("(")[1].split(")")[0].split(",")
        args_names = []
        for arg in args_list:
            args_names.append(arg.split(":")[0].strip())

        valid_args[valid_tools[i]] = args_names

    print("VALID_TOOLS: ", valid_tools)
    print("VALID_ARGS: ", valid_args)
    return valid_tools, valid_args


VALID_TOOLS, VALID_ARGS = get_valid_tools_and_args()
