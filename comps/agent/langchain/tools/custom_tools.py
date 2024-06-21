# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from typing import List

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.agent_toolkits.load_tools import load_tools


def get_duckduckgo_search(k: int = 10):
    from langchain_community.tools import DuckDuckGoSearchRun

    return DuckDuckGoSearchRun(k=k)


def get_profit_calculator():
    class CalculatorInput(BaseModel):
        principle: int = Field(description="the money invested in the beginning.")
        sold: int = Field(description="the money earned after selling the investment.")

    def profit_calculator(principle: float, sold: float) -> float:
        return "{:.2f}".format((sold - principle) / principle * 100)

    return StructuredTool(
        name="profit_calculator",
        description="Calculates profit percentage from principle and sold values.",
        func=profit_calculator,
        args_schema=CalculatorInput,
    )


def get_interest_calculator():
    class CalculatorInput(BaseModel):
        annual_rate: int = Field(description="the annual interest rate.")
        principle: int = Field(description="the principle amount.")

    def interest_calculator(annual_rate: float, principle: float):
        return "{:.2f}".format(principle * annual_rate / 12)

    return StructuredTool(
        name="interest_calculator",
        description="Calculates the interest for a given annual rate and principle.",
        func=interest_calculator,
        args_schema=CalculatorInput,
    )


def get_ticker_lookup():
    class TickerLookupInput(BaseModel):
        entity: str = Field(description="The name of the stock, ETF, or index.")

    def ticker_lookup(entity: str) -> str:
        import requests

        TICKER_DICT = {
            "foghorn therapeutics": "FHTX",
        }

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

    return StructuredTool(
        name="ticker_lookup",
        description="This function returns the ticker symbol for a stock, ETF, or an index.",
        func=ticker_lookup,
        args_schema=TickerLookupInput,
    )


def get_trade_info():
    class TradeInfoInput(BaseModel):
        ticker: str = Field(description="the stock or index ticker symbol. ONLY one ticker is allowed.")
        info_to_seek: str = Field(
            description="can only be one of the following: open_price, close_price, low_price, high_price, average_price, trading_volume, num_transaction."
        )
        from_date: str = Field(description="the start date for the aggregates, must be in the YYYY-MM-DD format.")
        span: str = Field(
            description="the time span for the aggregates, can be 'day', 'week', 'month', 'quarter' or 'year."
        )

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

    def get_trade_info(ticker: str, info_to_seek: str, from_date: str = None, span: str = "day") -> str:
        import datetime

        if "Ticker lookup tool cannot find ticker for" in ticker:
            return "Upstream ticker lookup failed, so cannot get aggregates."
        else:
            if from_date is None:
                from_date = datetime.datetime.now().strftime("%Y-%m-%d")
            aggregates = get_aggregates_for_ticker(ticker, from_date, span)
            if aggregates:
                return extract_info_from_aggregates(aggregates, info_to_seek)
            else:
                return "Cannot find trade info for ticker {}".format(ticker)

    return StructuredTool(
        name="get_trade_info",
        description="This function returns a single piece of information from the trading aggregates for a SINGLE stock or index ticker. The information can be open_price, close_price, low_price, high_price, average_price, trading_volume, num_transaction. No other info can be obtained with this tool.",
        func=get_trade_info,
        args_schema=TradeInfoInput,
    )


def tools_descriptions():
    tools = []
    try:
        tools += load_tools(
            [
                "google-finance",
            ]
        )
    except:
        pass
    tools += [
        get_duckduckgo_search(),
        # get_interest_calculator(),
        # get_profit_calculator(),
        get_ticker_lookup(),
        # get_trade_info(),
    ]
    return tools
