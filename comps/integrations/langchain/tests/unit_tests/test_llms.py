# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test OPEA Chat API wrapper."""

from langchain_opea import OPEALLM

OPEA_API_BASE = "http://localhost:9009/v1"
OPEA_API_KEY = "my_secret_value"
MODEL_NAME = "Intel/neural-chat-7b-v3-3"


def test_initialization() -> None:
    """Test integration initialization."""
    OPEALLM(opea_api_base=OPEA_API_BASE, opea_api_key=OPEA_API_KEY, model_name=MODEL_NAME)


def test_model_params() -> None:
    # Test standard tracing params
    llm = OPEALLM(opea_api_base=OPEA_API_BASE, opea_api_key=OPEA_API_KEY, model_name=MODEL_NAME)
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "opea",
        "ls_model_type": "llm",
        "ls_model_name": MODEL_NAME,
        "ls_temperature": 0.7,
        "ls_max_tokens": 256,
    }


def test_initialize_opeallm_bad_path_without_api_base() -> None:
    try:
        OPEALLM(opea_api_key=OPEA_API_KEY, model_name=MODEL_NAME)
    except ValueError as e:
        assert "opea_api_base" in e.__str__()


def test_initialize_opeallm_bad_path_without_api_key() -> None:
    try:
        OPEALLM(opea_api_base=OPEA_API_BASE, model_name=MODEL_NAME)
    except ValueError as e:
        assert "opea_api_key" in e.__str__()
