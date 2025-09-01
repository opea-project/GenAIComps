# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from bson.objectid import ObjectId

from comps.cores.storages import opea_store

dummy_doc = {"text": "mock data"}


class MockAsyncCursor:
    def __init__(self, docs):
        self.docs = docs
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.docs):
            raise StopAsyncIteration
        doc = self.docs[self.index]
        self.index += 1
        return doc


class MockSortCursor:
    def __init__(self, docs):
        self.docs = docs

    def sort(self, *args, **kwargs):
        return self

    async def to_list(self, length):
        return self.docs


class TestMongoDBStore(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.config = {
            "MONGO_HOST": "localhost",
            "MONGO_PORT": 27017,
            "OPEA_STORE_NAME": "mongodb",
            "DB_NAME": "test_db",
            "COLLECTION_NAME": "test_collection",
            "user": "test_user",
        }
        # patcher = patch("motor.motor_asyncio.AsyncIOMotorClient")
        patcher = patch("comps.cores.storages.mongodb.motor.AsyncIOMotorClient")
        self.addCleanup(patcher.stop)
        self.MockClient = patcher.start()

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_collection.count_documents = MagicMock(return_value=1)
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        self.MockClient.return_value = mock_client

        self.store = opea_store(name="mongodb", description="test", config=self.config)
        self.store.collection = mock_collection

    def test_health_check_success(self):
        self.store.collection.count_documents.return_value = 1
        result = self.store.health_check()
        self.assertTrue(result)

    def test_health_check_failure(self):
        self.store.collection.count_documents.side_effect = Exception("failed")
        result = self.store.health_check()
        self.assertFalse(result)

    async def test_asave_document(self):
        mock_id = ObjectId("60dbf3a1fc13ae1a3b000000")
        self.store.collection.insert_one.return_value.inserted_id = mock_id
        result = await self.store.asave_document(dummy_doc)
        self.assertEqual(result, str(mock_id))

    async def test_asave_documents(self):
        self.store.collection.insert_many.return_value.inserted_ids = [ObjectId()]
        docs = [dummy_doc]
        result = await self.store.asave_documents(docs)
        self.assertTrue(isinstance(result, str))

    async def test_aupdate_document(self):
        self.store.collection.update_one.return_value.modified_count = 1
        doc = {"doc_id": str(ObjectId()), "data": dummy_doc}
        result = await self.store.aupdate_document(doc)
        self.assertTrue(result)

    async def test_aupdate_documents(self):
        self.store.collection.update_one.return_value.modified_count = 1
        docs = [{"doc_id": str(ObjectId()), "data": dummy_doc}]
        result = await self.store.aupdate_documents(docs)
        self.assertTrue(result)

    async def test_aget_document_by_id(self):
        id = ObjectId()
        self.store.collection.find_one.return_value = {"_id": id, "data": {"text": "mock"}}
        result = await self.store.aget_document_by_id(str(id))
        self.assertEqual(result, {"_id": id, "data": {"text": "mock"}})

    async def test_aget_documents_by_ids(self):
        mock_id = ObjectId("60dbf3a1fc13ae1a3b000000")
        self.store.collection.find_one.return_value = {"_id": mock_id, "data": {"text": "mock"}}
        result = await self.store.aget_documents_by_ids([str(mock_id)])
        self.assertEqual(result, [{"_id": mock_id, "data": {"text": "mock"}}])

    async def test_adelete_document(self):
        self.store.collection.delete_one.return_value.deleted_count = 1
        result = await self.store.adelete_document(str(ObjectId()))
        self.assertTrue(result)

    async def test_adelete_documents(self):
        self.store.collection.delete_many.return_value.deleted_count = 1
        result = await self.store.adelete_documents([str(ObjectId())])
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
