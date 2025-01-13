# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test embedding model integration."""

from langchain_opea.embeddings import OPEAEmbeddings

OPEA_API_BASE = "http://localhost:9009/v1"
OPEA_API_KEY = "my_secret_value"
MODEL_NAME = "Intel/neural-chat-7b-v3-3"


def test_initialization() -> None:
    """Test embedding model initialization."""
    OPEAEmbeddings(
        opea_api_base=OPEA_API_BASE,
        opea_api_key=OPEA_API_KEY,
        model_name=MODEL_NAME,
    )
