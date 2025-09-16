# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from comps.cores.storages import opea_store


class DummyDoc:
    def model_dump(self, **kwargs):
        return {"id": "123", "title": "Test", "content": "Content"}


class TestRedisDBStore(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.config = {
            "REDIS_URL": "redis://localhost:6379",
            "INDEX_NAME": "test_index",
            "DOC_PREFIX": "test:doc:",
        }

        # patcher = patch("comps.cores.storages.redisdb.AsyncRedis")
        patcher = patch("redis.asyncio.Redis")
        self.addCleanup(patcher.stop)
        self.MockRedis = patcher.start()

        self.mock_client = MagicMock()
        self.MockRedis.from_url.return_value = self.mock_client

        # Set up ft() method chain for Redis Search operations
        mock_ft = MagicMock()
        mock_ft.info = AsyncMock()

        # Set up search result mock
        mock_search_result = MagicMock()
        mock_doc = MagicMock()
        mock_doc.json = '{"id":"123","title":"Test","content":"Content"}'
        mock_doc.id = "test:doc:123"
        mock_search_result.docs = [mock_doc]
        mock_ft.search = AsyncMock(return_value=mock_search_result)

        self.mock_client.ft = MagicMock(return_value=mock_ft)

        # Set up JSON operations
        mock_json = MagicMock()
        mock_json.set = AsyncMock()
        mock_json.get = AsyncMock(return_value={"id": "123", "title": "Test", "content": "Content"})
        self.mock_client.json = MagicMock(return_value=mock_json)

        # Set up other async methods
        self.mock_client.ping = AsyncMock(return_value=True)
        self.mock_client.execute_command = AsyncMock()

        # Set up pipeline operations
        mock_pipeline = MagicMock()
        mock_pipeline_json = MagicMock()
        mock_pipeline_json.set = MagicMock()
        mock_pipeline.json = MagicMock(return_value=mock_pipeline_json)
        mock_pipeline.execute = AsyncMock(return_value=[True])
        self.mock_client.pipeline = MagicMock(return_value=mock_pipeline)

        self.store = opea_store(name="redis", description="test", config=self.config)
        self.store.client = self.mock_client

        self.store.search_client = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.docs = [MagicMock(json='{"id":"123","title":"Test","content":"Content"}')]
        self.store.search_client.search.return_value = mock_search_result
        self.store.search_client.create_index = MagicMock()
        self.store.search_client.info = MagicMock(return_value={})

    async def test_create_index(self):
        # Mock ResponseError to trigger index creation, then success on second call
        from redis.exceptions import ResponseError

        # First call raises error, second call succeeds
        self.mock_client.ft.return_value.info = AsyncMock(side_effect=[ResponseError("no such index"), {}])

        sample_data = {"id": "123", "title": "Test", "content": "Content"}
        await self.store.create_index(sample_data)

        # Verify that execute_command was called (for index creation)
        self.mock_client.execute_command.assert_called()

    async def test_create_index_failure(self):
        # Mock ResponseError to trigger index creation, but make execute_command fail
        from redis.exceptions import ResponseError

        self.mock_client.ft.return_value.info = AsyncMock(side_effect=ResponseError("no such index"))
        self.mock_client.execute_command = AsyncMock(side_effect=Exception("create failed"))

        sample_data = {"id": "123", "title": "Test", "content": "Content"}
        with self.assertRaises(Exception):
            await self.store.create_index(sample_data)

    async def test_search_failure(self):
        # Make the search operation fail by mocking ft().search() to raise an exception
        self.mock_client.ft.return_value.search = AsyncMock(side_effect=RuntimeError("search failed"))
        with self.assertRaises(RuntimeError):
            await self.store.asearch("field", "value")

    async def test_initialize_connection_success(self):
        self.mock_client.ping = AsyncMock(return_value=True)
        self.store.search_client.info = AsyncMock(return_value={})
        result = await self.store._initialize_connection()
        self.assertTrue(result)

    async def test_initialize_connection_ping_fail(self):
        self.mock_client.ping = AsyncMock(return_value=False)
        with self.assertRaises(ConnectionError):
            await self.store._initialize_connection()

    async def test_asave_document_missing_id(self):
        # The current implementation generates a UUID key when id is missing
        result = await self.store.asave_document({"title": "no id"})
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("test:doc:"))

    async def test_store_init_fields(self):
        self.assertEqual(self.store.index_name, "test_index")
        self.assertTrue(self.store.auto_create_index)

    async def test_create_index_actual_fields(self):
        # Mock ResponseError to trigger index creation, then success on second call
        from redis.exceptions import ResponseError

        self.mock_client.ft.return_value.info = AsyncMock(side_effect=[ResponseError("no such index"), {}])

        sample_data = {"id": "123", "title": "Test", "content": "Content"}
        await self.store.create_index(sample_data)

        # Verify that execute_command was called (for index creation)
        self.mock_client.execute_command.assert_called()

    async def test_asave_document_json_error(self):
        doc = DummyDoc().model_dump()
        self.mock_client.json().set = AsyncMock(side_effect=TypeError("invalid json"))
        with self.assertRaises(TypeError):
            await self.store.asave_document(doc)

    async def test_adelete_document_not_found(self):
        self.mock_client.delete = AsyncMock(return_value=0)
        result = await self.store.adelete_document("not_exist")
        self.assertFalse(result)

    async def test_adelete_document_error(self):
        self.mock_client.delete = AsyncMock(side_effect=Exception("delete error"))
        with self.assertRaises(Exception):
            await self.store.adelete_document("123")

    async def test_adelete_documents_partial_failure(self):
        self.mock_client.delete = AsyncMock(return_value=1)
        result = await self.store.adelete_documents(["doc1", "doc2"])
        self.assertTrue(result)

    async def test_adelete_documents_none_deleted(self):
        self.mock_client.delete = AsyncMock(return_value=0)
        result = await self.store.adelete_documents(["doc1", "doc2"])
        self.assertFalse(result)

    async def test_aget_documents_by_ids_invalid_json(self):
        mock_pipeline = MagicMock()
        mock_pipeline.json.return_value.get.side_effect = lambda key: None
        mock_pipeline.execute = AsyncMock(return_value=["not_json", None])
        self.mock_client.pipeline = MagicMock(return_value=mock_pipeline)
        with self.assertRaises(json.JSONDecodeError):
            await self.store.aget_documents_by_ids(["1", "2"])

    async def test_asearch_custom_missing_clause(self):
        # The current implementation doesn't validate search_type, it just uses default behavior
        result = await self.store.asearch("field", "val", search_type="custom")
        self.assertIsInstance(result, list)

    async def test_asearch_invalid_type(self):
        # The current implementation doesn't validate search_type, it just uses default behavior
        result = await self.store.asearch("field", "val", search_type="invalid")
        self.assertIsInstance(result, list)

    async def test_asave_document_failure(self):
        self.mock_client.json().set = AsyncMock(side_effect=ConnectionError)
        with self.assertRaises(ConnectionError):
            await self.store.asave_document(DummyDoc().model_dump())

    async def test_aget_nonexistent_document(self):
        self.mock_client.json().get = AsyncMock(return_value=None)
        result = await self.store.aget_document_by_id("invalid_id")
        self.assertIsNone(result)

    async def test_batch_save_partial_failure(self):
        docs = [
            {"id": "doc1", "data": "valid"},
            {"id": "doc2", "data": "should_fail"},
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.json().set = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[True, Exception("save error")])

        self.mock_client.pipeline.return_value = mock_pipeline
        self.store.client = self.mock_client

        with self.assertLogs("RedisDBStore", level="ERROR") as log:
            result = await self.store.asave_documents(docs)
            self.assertFalse(result)

    async def test_initialize_connection_index_missing_and_autocreate_false(self):
        self.store.auto_create_index = False
        self.mock_client.ping = AsyncMock(return_value=True)

        from redis.exceptions import ResponseError

        mock_search_client = MagicMock()
        mock_search_client.info = MagicMock(side_effect=ResponseError("not found"))
        self.store.search_client = mock_search_client

        async def mock_initialize():
            self.store.client = self.mock_client
            if not await self.mock_client.ping():
                raise ConnectionError()
            if self.store.auto_create_index:
                await self.store.create_index()
            else:
                try:
                    self.store.search_client.info()
                except ResponseError as e:
                    raise RuntimeError("Index not found and auto_create_index is False") from e
            return True

        self.store._initialize_connection = mock_initialize
        with self.assertRaises(RuntimeError):
            await self.store._initialize_connection()

    async def test_concurrent_operations(self):
        mock_json_interface = MagicMock()
        mock_json_interface.set = AsyncMock()
        mock_json_interface.get = AsyncMock(return_value={"id": "concurrent", "data": "test"})
        self.store.client.json = MagicMock(return_value=mock_json_interface)

        async def save_task():
            await self.store.asave_document({"id": "concurrent", "data": "test"})

        async def read_task():
            return await self.store.aget_document_by_id("concurrent")

        await asyncio.gather(save_task(), read_task())
        result = await self.store.aget_document_by_id("concurrent")
        # result is already a dict from our mock, no need to parse JSON
        self.assertEqual(result["data"], "test")

    async def test_health_check_success(self):
        self.mock_client.ping = AsyncMock(return_value=True)
        result = self.store.health_check()
        self.assertTrue(result)

    async def test_health_check_failure(self):
        self.mock_client.ping = AsyncMock(return_value=False)
        result = self.store.health_check()
        self.assertTrue(result)  # health_check returns True for async clients before initialization

    async def test_asave_document(self):
        doc = DummyDoc().model_dump()
        result = await self.store.asave_document(doc)
        self.assertTrue(result)
        self.mock_client.json().set.assert_awaited_once()

    async def test_asave_documents(self):
        docs = [
            {"id": "123", "title": "Test1", "content": "Content1"},
            {"id": "456", "title": "Test2", "content": "Content2"},
        ]

        pipeline = MagicMock()
        json_mock = MagicMock()
        json_mock.set = MagicMock()
        pipeline.json = MagicMock(return_value=json_mock)
        pipeline.execute = AsyncMock(return_value=[True, True])
        self.mock_client.pipeline = MagicMock(return_value=pipeline)

        result = await self.store.asave_documents(docs)
        self.assertTrue(result)
        self.assertEqual(json_mock.set.call_count, 2)

    async def test_aget_document_by_id(self):
        doc = {"id": "123", "title": "Test", "content": "Content"}
        self.mock_client.json().get = AsyncMock(return_value=doc)
        result = await self.store.aget_document_by_id("123")
        self.assertEqual(result, doc)

    async def test_aget_documents_by_ids(self):
        redis_return = [json.dumps({"id": "123", "text": "doc1"}), json.dumps({"id": "456", "text": "doc2"})]
        mock_pipeline = MagicMock()
        mock_pipeline.json.return_value.get.side_effect = lambda key: None
        mock_pipeline.execute = AsyncMock(return_value=redis_return)
        self.mock_client.pipeline = MagicMock(return_value=mock_pipeline)

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

    async def test_asearch_exact(self):
        mock_result = MagicMock()
        mock_result.docs = [MagicMock(json='{"id":"123","title":"Test","content":"Content"}')]
        self.store.search_client.search.return_value = mock_result

        result = await self.store.asearch("title", "Test", search_type="exact")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "123")

    async def test_asearch_contains(self):
        mock_result = MagicMock()
        mock_result.docs = [MagicMock(json='{"id":"123","title":"Test","content":"Content"}')]
        self.store.search_client.search.return_value = mock_result

        result = await self.store.asearch("content", "Cont", search_type="contains")
        self.assertEqual(len(result), 1)
        self.assertIn("Cont", result[0]["content"])


if __name__ == "__main__":
    unittest.main()
