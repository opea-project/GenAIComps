# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

host_ip = os.getenv("host_ip")
# Neo4J configuration
NEO4J_URL = os.getenv("NEO4J_URL", f"bolt://{host_ip}:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jtest")

# LLM/Embedding endpoints
TGI_LLM_ENDPOINT = os.getenv("TGI_LLM_ENDPOINT", f"http://{host_ip}:6005")
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT ", f"http://{host_ip}:6006")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")

LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "meta-llama/Meta-Llama-3.1-70B-Instruct")
MAX_INPUT_LEN = os.getenv("MAX_INPUT_LEN", "8192")
MAX_OUTPUT_TOKENS = os.getenv("MAX_OUTPUT_TOKENS", "1024")
