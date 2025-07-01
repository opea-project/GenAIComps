# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from redis.asyncio import Redis as AsyncRedis

from comps.cores.storages import opea_store


class DummyDoc:
    def model_dump(self, **kwargs):
        return {"text": "mock data"}


class TestRedisDBStore(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.config = {
            "REDIS_URL": "redis://localhost:6379",
            "INDEX_NAME": "test_index",
            "DOC_PREFIX": "test_doc:",
            "AUTO_CREATE_INDEX": True,
        }

        # Patch the Redis client constructor
        self.patcher = patch("comps.cores.storages.redisdb.AsyncRedis")
        self.MockRedis = self.patcher.start()
        self.addCleanup(self.patcher.stop)

        # Create main client mocks
        self.mock_client = AsyncMock(spec=AsyncRedis)
        self.mock_json_client = AsyncMock()
        self.mock_index = AsyncMock()

        # Basic Redis client behaviors
        self.mock_client.ping = AsyncMock(return_value=True)
        self.mock_client.json.return_value = self.mock_json_client
        self.mock_client.ft.return_value = self.mock_index
        self.mock_client.delete = AsyncMock(return_value=1)
        self.MockRedis.from_url.return_value = self.mock_client

        # ✅ Mock pipeline behavior
        self.mock_pipeline = AsyncMock()
        self.mock_pipeline.execute = AsyncMock(return_value=[True, True])

        # ✅ Mock pipeline.json() behavior
        self.mock_pipeline_json = MagicMock()
        self.mock_pipeline_json.set = MagicMock(return_value=True)
        self.mock_pipeline_json.get = MagicMock(return_value={"id": "123", "text": "mock"})
        self.mock_pipeline_json.delete = MagicMock(return_value=True)

        # ✅ pipeline.json should be a callable returning mock_pipeline_json
        self.mock_pipeline.json = MagicMock(return_value=self.mock_pipeline_json)

        # ✅ pipeline itself
        self.mock_client.pipeline.return_value = self.mock_pipeline

        # Initialize store and inject mocks
        self.store = opea_store(name="redis", description="test", config=self.config)
        self.store.client = self.mock_client
        self.store.index = self.mock_index

    async def test_health_check_success(self):
        result = await self.store.health_check()
        self.assertTrue(result)

    async def test_health_check_failure(self):
        self.mock_client.ping.side_effect = Exception("Connection failed")
        result = await self.store.health_check()
        self.assertFalse(result)

    async def test_asave_document(self):
        doc = {"id": "123", "text": "test content"}
        self.mock_json_client.set = AsyncMock(return_value=True)
        result = await self.store.asave_document(doc)
        self.assertTrue(result)

    async def test_asave_documents(self):
        docs = [{"id": "123", "text": "doc1"}, {"id": "456", "text": "doc2"}]
        self.mock_pipeline.execute = AsyncMock(return_value=[True, True])
        result = await self.store.asave_documents(docs)
        self.assertTrue(result)

    async def test_aupdate_document(self):
        doc = {"id": "123", "text": "updated content"}
        self.mock_json_client.set = AsyncMock(return_value=True)
        result = await self.store.aupdate_document(doc)
        self.assertTrue(result)

    async def test_aupdate_documents(self):
        docs = [{"id": "123", "text": "doc1"}, {"id": "456", "text": "doc2"}]

        # Prepare mock responses
        self.mock_pipeline.execute = AsyncMock(return_value=[True, True])
        self.mock_pipeline_json.set = MagicMock(return_value=True)

        result = await self.store.aupdate_documents(docs)

        self.assertTrue(result)
        self.assertEqual(self.mock_pipeline_json.set.call_count, 2)
        self.mock_pipeline.execute.assert_awaited_once()

    async def test_aget_document_by_id(self):
        expected = {"id": "123", "text": "test content"}
        self.mock_json_client.get = AsyncMock(return_value=expected)
        result = await self.store.aget_document_by_id("123")
        self.assertEqual(result, expected)

    async def test_aget_documents_by_ids(self):
        redis_return = [json.dumps({"id": "123", "text": "doc1"}), json.dumps({"id": "456", "text": "doc2"})]

        mock_pipeline = MagicMock()
        mock_pipeline.json.return_value.get.side_effect = lambda key: None
        mock_pipeline.execute = AsyncMock(return_value=redis_return)

        self.store.client.pipeline = MagicMock(return_value=mock_pipeline)

        expected = [{"id": "123", "text": "doc1"}, {"id": "456", "text": "doc2"}]

        result = await self.store.aget_documents_by_ids(["123", "456"])
        self.assertEqual(result, expected)

    async def test_adelete_document(self):
        self.mock_client.delete = AsyncMock(return_value=1)
        result = await self.store.adelete_document("123")
        self.assertTrue(result)

    async def test_adelete_documents(self):
        self.mock_client.delete = AsyncMock(return_value=2)
        result = await self.store.adelete_documents(["123", "456"])
        self.assertTrue(result)

    async def test_asearch(self):
        # Mock search results
        mock_result = MagicMock()
        mock_doc = MagicMock()
        mock_doc.json = '{"id": "123", "title": "test", "content": "search content"}'
        mock_result.docs = [mock_doc]

        self.mock_index.search = AsyncMock(return_value=mock_result)

        # Execute search
        results = await self.store.asearch("content", "search")

        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "123")
        self.assertEqual(results[0]["title"], "test")

    async def test_asearch_failure(self):
        self.mock_index.search.side_effect = Exception("Search failed")
        with self.assertRaises(Exception):
            await self.store.asearch("content", "search")


if __name__ == "__main__":
    unittest.main()
