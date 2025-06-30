# Copyright (C) 2025 RedisDB Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any, Dict, List, Optional, Union

import redis
from redis.asyncio import Redis as AsyncRedis
from redis.commands.search.field import TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from ..common.storage import OpeaStore
from ..mega.logger import CustomLogger

logger = CustomLogger("RedisDBStore")


def escape_redis_query_value(value: str) -> str:
    # Escape special characters in Redis Search query syntax to avoid syntax errors
    special_chars = r'(){}[]|&!@~":'
    for ch in special_chars:
        value = value.replace(ch, f"\\{ch}")
    return value


class RedisDBStore(OpeaStore):
    """A concrete implementation of OpeaStore for Redis with search capabilities."""

    def __init__(self, name: str, description: str = "", config: dict = {}):
        """Initializes the RedisDBStore with the given configuration.

        Args:
            name (str): The name of the component.
            description (str): A brief description of the component.
            config (dict): Configuration parameters including:
                - REDIS_URL: Complete Redis URL (redis://...)
                - INDEX_NAME: Name of the Redis search index
                - DOC_PREFIX: Prefix for document keys (default: "doc:")
                - AUTO_CREATE_INDEX: Whether to auto-create the index if missing (default: True)
        """
        super().__init__(name, description, config)

        self.redis_url: str = config.get("REDIS_URL", "redis://localhost:6379")
        self.index_name: str = config.get("INDEX_NAME", "opea:index")
        self.doc_prefix: str = config.get("DOC_PREFIX", "doc:")
        self.auto_create_index: bool = config.get("AUTO_CREATE_INDEX", True)

        self.client: Optional[AsyncRedis] = None
        self.index = None

    async def initialize(self) -> bool:
        """Initialize the Redis connection and index (should be called after constructor)."""
        try:
            self.client = AsyncRedis.from_url(self.redis_url, decode_responses=True)

            if not await self.client.ping():
                raise ConnectionError(f"Failed to connect to Redis at {self.redis_url}")

            logger.info(f"Connected to Redis at {self.redis_url}")

            self.index = self.client.ft(self.index_name)

            try:
                await self.index.info()
                logger.debug(f"Using existing index: {self.index_name}")
            except redis.exceptions.ResponseError:
                if self.auto_create_index:
                    await self.create_index()
                else:
                    raise RuntimeError(f"Index '{self.index_name}' does not exist and auto-creation is disabled")

            return True
        except Exception as e:
            logger.error(f"Failed to initialize RedisDBStore: {e}")
            raise

    async def create_index(self) -> None:
        """Creates the Redis search index with flexible schema."""
        try:
            schema = (
                TagField("$.id", as_name="id"),
                TextField("$.title", as_name="title"),
                TextField("$.content", as_name="content"),
            )

            definition = IndexDefinition(prefix=[self.doc_prefix], index_type=IndexType.JSON)

            await self.index.create_index(schema, definition=definition)
            logger.info(f"Created Redis index '{self.index_name}' with prefix '{self.doc_prefix}'")
        except Exception as e:
            logger.error(f"Failed to create index '{self.index_name}': {e}")
            raise

    async def close(self) -> None:
        """Close the Redis connection cleanly."""
        try:
            if self.client:
                await self.client.close()
                logger.info("Closed Redis connection")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

    async def health_check(self) -> bool:
        """Performs a health check on the Redis connection."""
        try:
            if self.client is None:
                logger.error("Redis client not initialized")
                return False
            return await self.client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def save_document(self, doc: dict, **kwargs) -> bool:
        """Saves a single document to Redis as JSON."""
        try:
            if "id" not in doc:
                raise ValueError("Document must contain 'id' field")

            key = f"{self.doc_prefix}{doc['id']}"
            await self.client.json().set(key, "$", doc)
            return True
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            raise

    async def save_documents(self, docs: List[dict], **kwargs) -> bool:
        """Saves multiple documents to Redis using pipeline."""
        try:
            if not docs:
                return True

            pipeline = self.client.pipeline()
            for doc in docs:
                if "id" not in doc:
                    raise ValueError("All documents must contain 'id' field")

                key = f"{self.doc_prefix}{doc['id']}"
                pipeline.json().set(key, "$", doc)

            results = await pipeline.execute()
            success = all(res for res in results)
            if not success:
                logger.warning("Not all documents saved successfully in pipeline")
            return success
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
            raise

    async def update_document(self, doc: dict, **kwargs) -> bool:
        """Updates a document in Redis (same as save)."""
        return await self.save_document(doc, **kwargs)

    async def update_documents(self, docs: List[dict], **kwargs) -> bool:
        """Updates multiple documents in Redis (same as save)."""
        return await self.save_documents(docs, **kwargs)

    async def get_document_by_id(self, id: str, **kwargs) -> Optional[dict]:
        """Retrieves a document by its ID."""
        try:
            key = f"{self.doc_prefix}{id}"
            result = await self.client.json().get(key)
            return result if result else None
        except Exception as e:
            logger.error(f"Failed to get document by ID {id}: {e}")
            return None

    async def get_documents_by_ids(self, ids: List[str], **kwargs) -> List[dict]:
        """Retrieves multiple documents by their IDs."""
        try:
            if not ids:
                return []

            keys = [f"{self.doc_prefix}{id}" for id in ids]
            pipeline = self.client.pipeline()
            for key in keys:
                pipeline.json().get(key)

            results = await pipeline.execute()
            return [res for res in results if res]
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            raise

    async def delete_document(self, id: str, **kwargs) -> bool:
        """Deletes a document by ID."""
        try:
            key = f"{self.doc_prefix}{id}"
            deleted = await self.client.delete(key)
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete document {id}: {e}")
            raise

    async def delete_documents(self, ids: List[str], **kwargs) -> bool:
        """Deletes multiple documents by ID."""
        try:
            if not ids:
                return True
            keys = [f"{self.doc_prefix}{id}" for id in ids]
            deleted = await self.client.delete(*keys)
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise

    async def search(self, key: str, value: Any, search_type: str = "exact", **kwargs) -> List[dict]:
        """Search for documents by key and value using Redis Search."""
        try:
            value_str = str(value)
            value_esc = escape_redis_query_value(value_str)

            if search_type == "exact":
                query_str = f"@{key}:{{{value_esc}}}"
            elif search_type == "contains":
                query_str = f"@{key}:*{value_esc}*"
            elif search_type == "starts_with":
                query_str = f"@{key}:{value_esc}*"
            elif search_type == "ends_with":
                query_str = f"@{key}:*{value_esc}"
            elif search_type == "regex":
                return await self._regex_search(key, value_str)
            elif search_type == "custom":
                if "filter_clause" not in kwargs:
                    raise ValueError("Custom filter clause required for 'custom' search type")
                query_str = kwargs["filter_clause"]
            else:
                raise ValueError(f"Unsupported search type: {search_type}")

            query = Query(query_str)
            result = await self.index.search(query)

            return [json.loads(doc.json) for doc in result.docs]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def _regex_search(self, key: str, pattern: str) -> List[dict]:
        """Fallback regex search for unsupported regex in RedisSearch."""
        try:
            cursor = 0
            docs = []
            while True:
                cursor, keys = await self.client.scan(cursor, match=f"{self.doc_prefix}*")
                for key_name in keys:
                    doc = await self.client.json().get(key_name)
                    if doc and key in doc and re.search(pattern, str(doc[key])):
                        docs.append(doc)
                if cursor == 0:
                    break
            return docs
        except Exception as e:
            logger.error(f"Regex search failed: {e}")
            raise
