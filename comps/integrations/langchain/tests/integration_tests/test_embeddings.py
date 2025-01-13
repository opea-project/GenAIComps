# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test OPEA embeddings."""

from typing import Type

from langchain_opea.embeddings import OPEAEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

OPEA_API_BASE = "http://localhost:6006/v1"
OPEA_API_KEY = "my_secret_value"
MODEL_NAME = "BAAI/bge-large-en-v1.5"


class TestOPEAEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[OPEAEmbeddings]:
        return OPEAEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "opea_api_base": OPEA_API_BASE,
            "opea_api_key": OPEA_API_KEY,
            "model_name": MODEL_NAME,
        }
