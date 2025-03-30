import pytest
from unittest.mock import MagicMock, patch
from comps.cores.storages.arangodb import ArangoDBStore

@pytest.fixture
def mock_arangodb_store():
    """Fixture to create a mock instance of ArangoDBStore."""
    with patch('comps.cores.storages.arangodb.ArangoDBClient') as MockClient:
        mock_client = MockClient.return_value
        store = ArangoDBStore()
        store.client = mock_client
        yield store

def test_save_document(mock_arangodb_store):
    """Test saving a document to the database."""
    mock_arangodb_store.client.insert_document.return_value = {"_key": "123"}
    document = {"name": "test"}
    result = mock_arangodb_store.save(document)
    assert result["_key"] == "123"
    mock_arangodb_store.client.insert_document.assert_called_once_with(document)

def test_get_document(mock_arangodb_store):
    """Test retrieving a document from the database."""
    mock_arangodb_store.client.get_document.return_value = {"_key": "123", "name": "test"}
    result = mock_arangodb_store.get("123")
    assert result["name"] == "test"
    mock_arangodb_store.client.get_document.assert_called_once_with("123")

def test_update_document(mock_arangodb_store):
    """Test updating a document in the database."""
    mock_arangodb_store.client.update_document.return_value = {"_key": "123", "name": "updated"}
    document = {"_key": "123", "name": "updated"}
    result = mock_arangodb_store.update(document)
    assert result["name"] == "updated"
    mock_arangodb_store.client.update_document.assert_called_once_with(document)

def test_delete_document(mock_arangodb_store):
    """Test deleting a document from the database."""
    mock_arangodb_store.client.delete_document.return_value = True
    result = mock_arangodb_store.delete("123")
    assert result is True
    mock_arangodb_store.client.delete_document.assert_called_once_with("123")

def test_initialize_connection():
    """Test initializing the database connection."""
    with patch('comps.cores.storages.arangodb.ArangoDBClient') as MockClient:
        mock_client = MockClient.return_value
        store = ArangoDBStore()
        store.initialize_connection("http://localhost:8529", "test_db", "root", "password")
        mock_client.connect.assert_called_once_with(
            url="http://localhost:8529",
            database="test_db",
            username="root",
            password="password"
        )