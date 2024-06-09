# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os


# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# Embedding dimension
EMBED_DIMENSION = os.getenv("EMBED_DIMENSION", 768)

# Qdrant Connection Information
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))


def format_qdrant_conn_from_env():
    qdrant_url = os.getenv("QDRANT_URL", None)
    if qdrant_url:
        return qdrant_url
    else:
        start = "http://"
        return start + f"{QDRANT_HOST}:{QDRANT_PORT}"


QDRANT_URL = format_qdrant_conn_from_env()

# Vector Index Configuration
INDEX_NAME = os.getenv("INDEX_NAME", "rag-qdrant")

