# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os


#######################################################
#                Common Parameters                    #
#######################################################
def get_boolean_env_var(var_name, default_value=False):
    """Retrieve the boolean value of an environment variable.

    Args:
    var_name (str): The name of the environment variable to retrieve.
    default_value (bool): The default value to return if the variable
    is not found.

    Returns:
    bool: The value of the environment variable, interpreted as a boolean.
    """
    true_values = {"true", "1", "t", "y", "yes"}
    false_values = {"false", "0", "f", "n", "no"}

    # Retrieve the environment variable's value
    value = os.getenv(var_name, "").lower()

    # Decide the boolean value based on the content of the string
    if value in true_values:
        return True
    elif value in false_values:
        return False
    else:
        return default_value


# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# Embedding endpoint
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "")
# LLM endpoint
LLM_ENDPOINT = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
LLM_ENDPOINT_NO_RAG = os.getenv("TGI_LLM_ENDPOINT_NO_RAG", "http://localhost:8081")


#######################################################
#                     Redis                           #
#######################################################
# Connection Information
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


def format_redis_conn_from_env():
    redis_url = os.getenv("REDIS_URL", None)
    if redis_url:
        return redis_url
    else:
        using_ssl = get_boolean_env_var("REDIS_SSL", False)
        start = "rediss://" if using_ssl else "redis://"

        # if using RBAC
        password = os.getenv("REDIS_PASSWORD", None)
        username = os.getenv("REDIS_USERNAME", "default")
        if password is not None:
            start += f"{username}:{password}@"

        return start + f"{REDIS_HOST}:{REDIS_PORT}"


REDIS_URL = format_redis_conn_from_env()

# Index Configuration
INDEX_NAME = os.getenv("INDEX_NAME", "rag_redis")
KEY_INDEX_NAME = os.getenv("KEY_INDEX_NAME", "file_keys")
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 600))
SEARCH_BATCH_SIZE = int(os.getenv("SEARCH_BATCH_SIZE", 10))

# Schema Configuration
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
REDIS_SCHEMA = os.getenv("REDIS_SCHEMA", "redis_schema_multi.yml")
schema_path = os.path.join(parent_dir, REDIS_SCHEMA)
INDEX_SCHEMA = schema_path


#######################################################
#                     Milvus                          #
#######################################################
# Connection Information
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
INDEX_PARAMS = {"index_type": "FLAT", "metric_type": "IP", "params": {}}
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_milvus")

# Embedding Endpoint Configuration
os.environ["OPENAI_API_BASE"] = EMBEDDING_ENDPOINT
os.environ["OPENAI_API_KEY"] = "Dummy key"


#######################################################
#                     Neo4j                           #
#######################################################
# Neo4J configuration
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")


#######################################################
#                     Pinecone                        #
#######################################################
# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "xxx_xxx")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-test")

