# Copyright (C) 2025 RedisDB Inc.
# SPDX-License-Identifier: Apache-2.0
"""OPEA Storage Factory Module.

This module provides a factory pattern for creating and managing different storage backends
in the OPEA ecosystem. It supports multiple storage systems including MongoDB, ArangoDB,
and Redis, with configuration management through environment variables.

The factory pattern allows for easy switching between storage backends without changing
application code, making the system more flexible and maintainable.
"""

import os

from comps.cores.common.storage import OpeaStore

STORE_ID_COLS = {
    "mongodb": "_id",
    "arangodb": "_id",
    "redis": "id",
}


def get_store_name() -> str:
    """Retrieves the configured storage backend name from environment variables.

    This function reads the OPEA_STORE_NAME environment variable to determine
    which storage backend should be used by the application. The name is
    normalized to lowercase for consistency.

    Returns:
        str: The normalized storage backend name (e.g., "mongodb", "arangodb", "redis").

    Raises:
        Exception: If the OPEA_STORE_NAME environment variable is not set or is empty.

    Example:
        >>> os.environ['OPEA_STORE_NAME'] = 'arangodb'
        >>> get_store_name()
        'arangodb'
    """
    store_name = os.getenv("OPEA_STORE_NAME")
    if store_name is None:
        raise Exception(
            "Environment variable 'OPEA_STORE_NAME' is not set. "
            "Please configure it with a supported storage backend name (mongodb, arangodb, redis)."
        )
    if store_name not in STORE_ID_COLS.keys():
        raise Exception(
            f"Storage backend '{store_name}' is not supported. " f"Supported backends are: mongodb, arangodb, redis"
        )
    return store_name.lower()


def _get_store_cfg(user: str) -> dict:
    """Builds and returns the configuration dictionary for the specified storage backend.

    This internal function creates a configuration dictionary containing all necessary
    parameters for initializing the chosen storage backend. It reads environment
    variables with sensible defaults to configure the storage connection.

    Args:
        user (str): The username/identifier for whom the configuration is being generated.
                   This is used for user-scoped data isolation and access control.

    Returns:
        dict: A dictionary containing all configuration parameters required for the
              specified storage backend. The structure varies by backend type:

              - MongoDB: Contains MONGO_HOST, MONGO_PORT, DB_NAME, COLLECTION_NAME
              - ArangoDB: Contains ARANGODB_HOST, ARANGODB_USERNAME, ARANGODB_PASSWORD, etc.
              - Redis: Contains REDIS_URL, INDEX_NAME, DOC_PREFIX, AUTO_CREATE_INDEX

    Raises:
        Exception: If the configured storage backend is not supported.

    Note:
        This is an internal function and should not be called directly.
        Use get_store() instead for public API access.
    """
    name = get_store_name()

    # ArangoDB configuration with connection and authentication parameters
    if name == "arangodb":
        return {
            "user": user,
            "is_async": True,  # Default to async for better FastAPI integration
            "ARANGODB_HOST": os.getenv("ARANGODB_HOST", "http://localhost:8529"),
            "ARANGODB_USERNAME": os.getenv("ARANGODB_USERNAME", "root"),
            "ARANGODB_PASSWORD": os.getenv("ARANGODB_PASSWORD", ""),
            "ARANGODB_DB_NAME": os.getenv("ARANGODB_DB_NAME", "_system"),
            "ARANGODB_COLLECTION_NAME": os.getenv("ARANGODB_COLLECTION_NAME", "default"),
        }

    # MongoDB configuration with host, port, and database settings
    elif name == "mongodb":
        return {
            "user": user,
            "MONGO_HOST": os.getenv("MONGO_HOST", "localhost"),
            "MONGO_PORT": os.getenv("MONGO_PORT", 27017),
            "DB_NAME": os.getenv("DB_NAME", "OPEA"),
            "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "ChatHistory"),
        }

    # Redis configuration with URL and indexing parameters
    elif name == "redis":
        return {
            "user": user,
            "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379"),
            "INDEX_NAME": os.getenv("INDEX_NAME", "opea:index"),
            "DOC_PREFIX": os.getenv("DOC_PREFIX", "doc:"),
            "AUTO_CREATE_INDEX": os.getenv("AUTO_CREATE_INDEX", True),
        }

    # Future storage backends can be added here following the same pattern
    else:
        raise Exception(
            f"Storage backend '{name}' is not supported. " f"Supported backends are: mongodb, arangodb, redis"
        )


def get_store(user: str) -> OpeaStore:
    """Factory function to create and initialize a storage backend instance.

    This is the main entry point for obtaining a configured and connected storage
    instance. It uses the factory pattern to abstract storage backend selection
    and initialization, making it easy to switch between different storage systems.

    The function performs the following operations:
    1. Validates the user parameter
    2. Determines the storage backend from environment configuration
    3. Creates the appropriate storage instance with configuration
    4. Initializes the storage connection/database
    5. Performs a health check to ensure the connection is working

    Args:
        user (str): The username/identifier for whom the store is being initialized.
                   This is required for user-scoped data access and isolation.
                   Cannot be None or empty.

    Returns:
        OpeaStore: A fully initialized and health-checked instance of the requested
                  storage backend. The instance is ready for immediate use.

    Raises:
        Exception: If any of the following conditions occur:
                  - User information is not provided (None or empty)
                  - The configured storage backend is not supported
                  - Storage initialization fails
                  - Health check fails (cannot connect to storage)

    Example:
        >>> os.environ['OPEA_STORE_NAME'] = 'mongodb'
        >>> store = get_store('user_foo')
    """
    # Validate user parameter - required for all storage operations
    if not user:
        raise Exception(
            "User information is required to initialize the data store. " "Please provide a valid user identifier."
        )

    name = get_store_name()
    store_cfg = _get_store_cfg(user)

    store = None

    # Initialize MongoDB store with database setup
    if name == "mongodb":
        from comps.cores.storages.mongodb import MongoDBStore

        store = MongoDBStore(name, config=store_cfg)
        store._initialize_db()

    # Initialize ArangoDB store with connection setup
    elif name == "arangodb":
        from comps.cores.storages.arangodb import ArangoDBStore

        store = ArangoDBStore(name, config=store_cfg)
        # For async ArangoDB, initialization happens lazily in async methods
        if not store_cfg.get("is_async", store.IS_ASYNC_DEFAULT):
            store._initialize_connection_sync()

    # Initialize Redis store with connection setup
    elif name == "redis":
        from comps.cores.storages.redisdb import RedisDBStore

        store = RedisDBStore(name, config=store_cfg)
        # For async Redis, initialization happens lazily in async methods
        if not store_cfg.get("is_async", store.IS_ASYNC_DEFAULT):
            store._initialize_connection_sync()

    # Future storage backends can be added here.

    # Ensure we have a valid store instance
    if store is None:
        raise Exception(
            f"Storage backend '{name}' is not supported. " f"Supported backends are: mongodb, arangodb, redis"
        )

    # Verify the store is healthy and ready for use
    if store.health_check():
        return store
    else:
        raise Exception(
            f"Failed to establish a healthy connection to {name} storage backend. "
            f"Please check your configuration and ensure the storage service is running."
        )


def remove_db_private_cols(doc: dict) -> dict:
    """Removes private database fields from the document dictionary.

    This function cleans up a document dictionary by removing fields that are
    considered private or internal to the database system. This is useful for
    preparing documents for external consumption, such as API responses, where
    such internal fields should not be exposed.

    Args:
        doc (dict): The document dictionary from which private fields should be removed.

    Returns:
        dict: The cleaned document dictionary with private fields removed.

    Note:
        The specific private fields removed depend on the configured storage backend.
        Common fields include MongoDB's '_id' and ArangoDB's '_key', but this can vary.

    Example:
        >>> doc = {'_id': '123', 'name': 'Alice', '_key': 'abc'}
        >>> remove_db_private_cols(doc)
        {'name': 'Alice'}
    """
    store_name = get_store_name()
    private_fields = {
        "mongodb": ["_id"],
        "arangodb": ["_key", "_id", "_rev", "_oldRev"],
        "redis": [],  # Redis does not have standard private fields in documents
    }

    fields_to_remove = private_fields.get(store_name, [])
    for field in fields_to_remove:
        doc.pop(field, None)  # Remove field if it exists, ignore if not

    return doc


def prepersist(col_name: str, doc: dict) -> dict:
    """Formats the document's ID field to match store's requirements.

    Args:
        col_name (str): The name of the ID column.
        doc (dict): The document to be formatted.
    Returns:
        dict: The formatted document with the correct ID field.
    """
    store_name = get_store_name()
    if col_name and col_name in doc:
        doc[STORE_ID_COLS[store_name]] = doc.pop(col_name)
    return doc


def postget(col_name: str, doc: dict) -> dict:
    """Formats the document's ID field from store's requirements
    to the application's requirements.

    Args:
        col_name (str): The name of the ID column.
        doc (dict): The document to be formatted.
    Returns:
        dict: The formatted document with the correct ID field.
    """
    store_name = get_store_name()
    if col_name and STORE_ID_COLS[store_name] in doc:
        doc[col_name] = str(doc.pop(STORE_ID_COLS[store_name]))
    doc = remove_db_private_cols(doc)
    return doc


def get_id_col_name() -> str:
    """Retrieves the ID column name for the configured storage backend.

    This function returns the appropriate ID column name based on the
    currently configured storage backend. It abstracts away the differences
    in ID field naming conventions across different storage systems.

    Returns:
        str: The ID column name used by the configured storage backend.

    Raises:
        Exception: If the configured storage backend is not supported.

    Example:
        >>> os.environ['OPEA_STORE_NAME'] = 'mongodb'
        >>> get_id_col_name()
        '_id'
    """
    store_name = get_store_name()
    if store_name in STORE_ID_COLS:
        return STORE_ID_COLS[store_name]
    else:
        raise Exception(
            f"Storage backend '{store_name}' is not supported. " f"Supported backends are: mongodb, arangodb, redis"
        )
