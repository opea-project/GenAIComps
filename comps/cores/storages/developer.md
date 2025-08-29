# Contributing to Storage

## Storage

`OpeaStore` is responsible for managing data persistence and retrieval. It is designed as a Client API, allowing for easy integration with other components and services. The storage system is built to be modular and extensible, enabling developers to add new features and capabilities as needed.

All storage components inherit from the `OpeaStore` class in `comps/cores/common/storage.py`. This class provides a common interface and base functionality for all storage implementations. The `OpeaStore` class is designed to be flexible and adaptable, allowing for different storage backends and configurations.

## Adding a New Storage Backend

To add a new storage backend, follow these steps:

1. Implement your storage class in `comps/cores/storages/your_storage.py`. It should inherit from `OpeaStore` in `comps/cores/common/storage.py`.
2. Configure the `opea_store()` function in `comps/cores/storages/__init__.py` to include your new storage backend.
3. Introduce any necessary data storage models to `comps/cores/storages/models.py`.

## Example

```python
from comps.cores.common.storage import OpeaStore

store = OpeaStore(name="arangodb")  # "redis", "mongodb", etc.

result = store.save_document({"foo": "bar"})

store.save_documents([{"foo": "baz"}, {"foo": "faz"}])

store.update_document({"_id": result["_id"], "foo": "bar!!!"})

store.update_documents(...)

result_2 = store.get_document_by_id(result["_id"])

assert result != result_2  # "foo" attribute now has different value

store.get_documents_by_ids([...])

result_3 = store.search(key="foo", value="bar!!!")[0]

assert result_2 == result_3

store.delete_document(result["_id"])

store.delete_documents([...])
```
