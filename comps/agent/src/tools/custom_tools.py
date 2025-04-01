# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
import random
from typing import Annotated
from collections import defaultdict
from datetime import datetime, date, timedelta
import pandas as pd

finnhub_client = None

try:
    if os.environ.get("FINNHUB_API_KEY") is None:
        print(
                "Please set the environment variable FINNHUB_API_KEY to use the Finnhub API."
        )
    else:
        import finnhub
        finnhub_client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])
        print("Finnhub client initialized")

except:
    pass


def get_company_profile(symbol: Annotated[str, "ticker symbol"]) -> str:
    """
    get a company's profile information
    """
    profile = finnhub_client.company_profile2(symbol=symbol)
    if not profile:
        return f"Failed to find company profile for symbol {symbol} from finnhub!"

    formatted_str = (
        "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. "
        "Incorporated and publicly traded since {ipo}, the company has established its reputation as "
        "one of the key players in the market. As of today, {name} has a market capitalization "
        "of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding."
        "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. "
        "As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive "
        "progress within the industry."
    ).format(**profile)

    return formatted_str


def get_company_news(
    symbol: Annotated[str, "ticker symbol"],
    start_date: Annotated[
        str,
        "start date of the search period for the company's basic financials, yyyy-mm-dd",
    ],
    end_date: Annotated[
        str,
        "end date of the search period for the company's basic financials, yyyy-mm-dd",
    ],
    max_news_num: Annotated[
        int, "maximum number of news to return, default to 10"
    ] = 10,
    ):
    """
    retrieve market news related to designated company
    """
    news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
    if len(news) == 0:
        print(f"No company news found for symbol {symbol} from finnhub!")
    news = [
        {
            "date": datetime.fromtimestamp(n["datetime"]).strftime("%Y%m%d%H%M%S"),
            "headline": n["headline"],
            "summary": n["summary"],
        }
        for n in news
    ]
    # Randomly select a subset of news if the number of news exceeds the maximum
    if len(news) > max_news_num:
        news = random.choices(news, k=max_news_num)
    news.sort(key=lambda x: x["date"])
    output = pd.DataFrame(news)

    return output.to_json(orient="split")

def get_current_date():
    return date.today().strftime("%Y-%m-%d")


# tool for unit test
def search_web(query: str) -> str:
    """Search the web knowledge for a given query."""
    ret_text = """
    The Linux Foundation AI & Data announced the Open Platform for Enterprise AI (OPEA) as its latest Sandbox Project.
    OPEA aims to accelerate secure, cost-effective generative AI (GenAI) deployments for businesses by driving interoperability across a diverse and heterogeneous ecosystem, starting with retrieval-augmented generation (RAG).
    """
    return ret_text


def search_weather(query: str) -> str:
    """Search the weather for a given query."""
    ret_text = """
    It's clear.
    """
    return ret_text

