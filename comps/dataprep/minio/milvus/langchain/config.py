# Copyright (c) 2015-2024 MinIO, Inc.
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
MOSEC_EMBEDDING_MODEL = os.environ.get("MOSEC_EMBEDDING_MODEL", "/home/user/bge-large-zh-v1.5")
MOSEC_EMBEDDING_ENDPOINT = os.environ.get("MOSEC_EMBEDDING_ENDPOINT", "")
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "False").lower() == "true"
MINIO_DOCUMENT_BUCKET = os.environ.get("MINIO_DOCUMENT_BUCKET", "document")
MINIO_WAREHOUSE_BUCKET = os.environ.get("MINIO_WAREHOUSE_BUCKET", "warehouse")
os.environ["OPENAI_API_BASE"] = MOSEC_EMBEDDING_ENDPOINT
os.environ["OPENAI_API_KEY"] = "Dummy key"
