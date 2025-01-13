# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test OPEALLM llm."""

from langchain_opea.llms import OPEALLM

OPEA_API_BASE = "http://localhost:9009/v1"
OPEA_API_KEY = "my_secret_value"
MODEL_NAME = "Intel/neural-chat-7b-v3-3"


def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OPEALLM(opea_api_base=OPEA_API_BASE, opea_api_key=OPEA_API_KEY, model_name=MODEL_NAME)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OPEALLM(opea_api_base=OPEA_API_BASE, opea_api_key=OPEA_API_KEY, model_name=MODEL_NAME)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_abatch() -> None:
    """Test streaming tokens from OPEALLM."""
    llm = OPEALLM(opea_api_base=OPEA_API_BASE, opea_api_key=OPEA_API_KEY, model_name=MODEL_NAME)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from OPEALLM."""
    llm = OPEALLM(opea_api_base=OPEA_API_BASE, opea_api_key=OPEA_API_KEY, model_name=MODEL_NAME)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]})
    for token in result:
        assert isinstance(token, str)


def test_batch() -> None:
    """Test batch tokens from OPEALLM."""
    llm = OPEALLM(opea_api_base=OPEA_API_BASE, opea_api_key=OPEA_API_KEY, model_name=MODEL_NAME)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from OPEALLM."""
    llm = OPEALLM(opea_api_base=OPEA_API_BASE, opea_api_key=OPEA_API_KEY, model_name=MODEL_NAME)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


def test_invoke() -> None:
    """Test invoke tokens from OPEALLM."""
    llm = OPEALLM(opea_api_base=OPEA_API_BASE, opea_api_key=OPEA_API_KEY, model_name=MODEL_NAME)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)
