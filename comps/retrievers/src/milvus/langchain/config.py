# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

# Local Embedding model
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "maidalun1020/bce-embedding-base_v1")
# TEI Embedding endpoints
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
# MILVUS configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_milvus")
# MOSEC configuration
MOSEC_EMBEDDING_MODEL = os.environ.get("MOSEC_EMBEDDING_MODEL", "/home/user/bce-embedding-base_v1")
MOSEC_EMBEDDING_ENDPOINT = os.environ.get("MOSEC_EMBEDDING_ENDPOINT", "")
os.environ["OPENAI_API_BASE"] = MOSEC_EMBEDDING_ENDPOINT
os.environ["OPENAI_API_KEY"] = "Dummy key"
