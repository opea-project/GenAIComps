# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from comps.cores.storages import opea_store


@pytest.fixture
def mock_arangodb_store():
    """Fixture to create a mock instance of ArangoDBStore."""
    with patch("arango.ArangoClient") as MockClient:
        mock_client = MockClient.return_value
        store = opea_store("arangodb", config={"client": mock_client})
        yield store


def test_crud(mock_arangodb_store):
    """Mock CRUD operations."""
    document = {"name": "test"}
    assert mock_arangodb_store.save_document(document)
    assert mock_arangodb_store.save_documents([document])
    assert mock_arangodb_store.update_document(document)
    assert mock_arangodb_store.update_documents([document])
    assert mock_arangodb_store.delete_document("123")
    assert mock_arangodb_store.delete_documents(["123"])
    assert mock_arangodb_store.get_document_by_id("123")
    assert mock_arangodb_store.get_documents_by_ids(["123"])


def test_initialize_connection():
    """Test initializing the database connection."""
    with patch("arango.ArangoClient") as MockClient:
        mock_client = MockClient.return_value
        store = opea_store("arangodb", config={"client": mock_client})
        assert store.client == mock_client
        assert store.db is not None
        assert store.collection is not None
