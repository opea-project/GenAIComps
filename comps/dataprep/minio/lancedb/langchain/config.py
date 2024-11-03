# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

# Local Embedding model
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "maidalun1020/bce-embedding-base_v1")
# TEI Embedding endpoints
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
# LanceDB configuration
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
os.environ["AWS_ENDPOINT"] = f"http://{MINIO_ENDPOINT}"
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["ALLOW_HTTP"] = str(MINIO_SECURE != "true").lower()
