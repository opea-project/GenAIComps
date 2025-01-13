# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test chat model integration."""

from langchain_opea.chat_models import ChatOPEA

OPEA_API_BASE = "http://localhost:9009/v1"
OPEA_API_KEY = "my_secret_value"
MODEL_NAME = "Intel/neural-chat-7b-v3-3"


def test_initialize_opea() -> None:
    llm = ChatOPEA(
        opea_api_base=OPEA_API_BASE,
        opea_api_key=OPEA_API_KEY,
        model_name=MODEL_NAME,
    )
    assert llm.opea_api_base == OPEA_API_BASE
    assert llm.opea_api_key.get_secret_value() == OPEA_API_KEY
    assert llm.model_name == MODEL_NAME


def test_initialize_more() -> None:
    llm = ChatOPEA(  # type: ignore[call-arg]
        opea_api_base=OPEA_API_BASE,
        opea_api_key=OPEA_API_KEY,
        model_name=MODEL_NAME,
        temperature=0,
        max_retries=3,
    )

    assert llm.opea_api_key.get_secret_value() == OPEA_API_KEY
    assert llm.opea_api_base == OPEA_API_BASE
    assert llm.model_name == MODEL_NAME
    assert llm.max_retries == 3
    assert llm.temperature == 0
