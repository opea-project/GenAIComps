# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

# ArangoDB configuration
ARANGO_URL = os.getenv("ARANGO_URL", "http://localhost:8529")
ARANGO_USERNAME = os.getenv("ARANGO_USERNAME", "root")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD", "test")
ARANGO_DB_NAME = os.getenv("ARANGO_DB_NAME", "_system")

# Text Generation Inference configuration
TGI_LLM_ENDPOINT = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")

# Text Embeddings Inference configuration
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
TEI_EMBED_MODEL = os.getenv("TEI_EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# OpenAI configuration (alternative to TGI & TEI)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
