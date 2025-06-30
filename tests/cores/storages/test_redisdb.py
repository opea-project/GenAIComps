# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest

from comps.cores.storages import opea_store


@pytest.fixture
def mock_redisdb_store():
    """Fixture to create a mock instance of RedisDBStore."""
    with patch("redis.asyncio.Redis.from_url") as mock_client_factory:
        mock_client = AsyncMock()
        mock_index = AsyncMock()

        # Simulate Redis client behavior
        mock_client.ping.return_value = True
        mock_client.ft.return_value = mock_index
        mock_index.info.return_value = {"index_name": "mock_index"}
        mock_index.search.return_value = AsyncMock(docs=[])

        mock_client_factory.return_value = mock_client

        config = {
            "REDIS_URL": "redis://mockhost:6379",
            "INDEX_NAME": "mock_index",
            "DOC_PREFIX": "mockdoc:",
            "AUTO_CREATE_INDEX": True,
            "client": mock_client,  # Optional direct client injection if supported
        }
        store = opea_store("redis", config=config)
        store.client = mock_client
        store.index = mock_index
        yield store


@pytest.mark.asyncio
async def test_crud_operations(mock_redisdb_store):
    """Test mocked Redis CRUD operations."""
    document = {"id": "1", "field": "value"}

    # Mock JSON interface
    mock_json = AsyncMock()
    mock_redisdb_store.client.json.return_value = mock_json
    mock_json.set.return_value = True
    mock_json.get.return_value = document
    mock_redisdb_store.client.delete.return_value = 1

    assert await mock_redisdb_store.save_document(document)
    assert await mock_redisdb_store.save_documents([document])
    assert await mock_redisdb_store.update_document(document)
    assert await mock_redisdb_store.update_documents([document])
    assert await mock_redisdb_store.delete_document("1")
    assert await mock_redisdb_store.delete_documents(["1"])
    assert await mock_redisdb_store.get_document_by_id("1")
    assert await mock_redisdb_store.get_documents_by_ids(["1"])


@pytest.mark.asyncio
async def test_initialize_connection():
    """Test Redis store initialization."""
    with patch("redis.asyncio.Redis.from_url") as mock_client_factory:
        mock_client = AsyncMock()
        mock_index = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.ft.return_value = mock_index
        mock_index.info.return_value = {"index_name": "mock_index"}

        mock_client_factory.return_value = mock_client

        config = {
            "REDIS_URL": "redis://mockhost:6379",
            "INDEX_NAME": "mock_index",
            "DOC_PREFIX": "mockdoc:",
            "AUTO_CREATE_INDEX": True,
            "client": mock_client,
        }
        store = opea_store("redis", config=config)
        assert store.client == mock_client
        assert store.index is not None
