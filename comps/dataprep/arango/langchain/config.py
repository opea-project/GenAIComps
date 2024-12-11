# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

# ArangoDB configuration
ARANGO_URL = os.getenv("ARANGO_URL", "http://localhost:8529")
ARANGO_USERNAME = os.getenv("ARANGO_USERNAME", "root")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD", "test")
ARANGO_DB_NAME = os.getenv("ARANGO_DB_NAME", "_system")

# ArangoDB graph configuration
USE_ONE_ENTITY_COLLECTION = True
INSERT_ASYNC = False
ARANGO_BATCH_SIZE = 1000
INCLUDE_SOURCE = True

# Text Generation Inference configuration
TGI_LLM_ENDPOINT = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")

# Text Embeddings Inference configuration
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
TEI_EMBED_MODEL = os.getenv("TEI_EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# OpenAI configuration (alternative to TGI & TEI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_EMBED_DIMENSIONS = os.getenv("OPENAI_EMBED_DIMENSIONS", 512)

# LLMGraphTransformer configuration
ALLOWED_NODES = [] # ["Person", "Organization"]
ALLOWED_RELATIONSHIPS = [] # [("Person", "knows", "Person"), ("Person", "works_at", "Organization")]
NODE_PROPERTIES = False # ["description"]
RELATIONSHIP_PROPERTIES = False # ["description"]

SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "./prompt.txt")