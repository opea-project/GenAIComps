# Copyright (C) 2025 RedisDB Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from typing import Any, Dict, List, Optional

from redis.commands.search.field import NumericField, TagField, TextField
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from ..common.storage import OpeaStore
from ..mega.logger import CustomLogger

logger = CustomLogger("RedisDBStore")


class RedisQueryBuilder:
    """Helper class for building Redis search queries."""

    @staticmethod
    def escape_query_value(value: str) -> str:
        """Escape special characters in Redis Search query syntax to avoid syntax errors."""
        special_chars = r'(){}[]|&!@~":'
        for ch in special_chars:
            value = value.replace(ch, f"\\{ch}")
        return value

    @staticmethod
    def build_search_query(key: str, value: Any, search_type: str = "exact") -> str:
        """Build a Redis search query string based on value type and search type."""
        # Convert dots to underscores for field names
        key = key.replace(".", "_")

        if isinstance(value, bool):
            # For boolean values (TAG fields), use curly braces
            return f"@{key}:{{{str(value).lower()}}}"
        elif isinstance(value, str):
            # For string values, handle search_type
            if search_type == "exact":
                return f'@{key}:"{value}"'
            elif search_type == "contains":
                return f"@{key}:*{value}*"
            else:
                return f"@{key}:{value}"
        else:
            # For numeric values
            return f"@{key}:{value}"


class RedisSchemaGenerator:
    """Helper class for generating Redis search schemas from sample data."""

    @staticmethod
    def generate_schema(data_obj: Dict, current_path: str = "$", schema_fields: Optional[List] = None) -> tuple:
        """Recursively traverses a dictionary to generate a RediSearch schema.

        Args:
            data_obj: The data object to analyze
            current_path: Current JSONPath being processed
            schema_fields: Accumulated schema fields

        Returns:
            Tuple of schema fields for Redis index creation
        """
        if schema_fields is None:
            schema_fields = []

        for key, value in data_obj.items():
            # Construct the JSONPath for the current key
            new_path = f"{current_path}.{key}"

            # Create a clean alias for querying (e.g., 'address.city' -> 'address_city')
            alias = new_path[2:].replace(".", "_")

            if isinstance(value, dict):
                RedisSchemaGenerator.generate_schema(value, new_path, schema_fields)
            elif isinstance(value, bool):
                schema_fields.append(TagField(new_path, as_name=alias, sortable=True))
            elif isinstance(value, str):
                schema_fields.append(TextField(new_path, as_name=alias, sortable=True))
            elif isinstance(value, (int, float)):
                schema_fields.append(NumericField(new_path, as_name=alias, sortable=True))
            # Skip lists and None values for now
            elif isinstance(value, list) or value is None:
                pass

        return tuple(schema_fields)


class RedisDBStore(OpeaStore):
    """A concrete implementation of OpeaStore for Redis with search capabilities."""

    # Class constants
    IS_ASYNC_DEFAULT = True
    DEFAULT_REDIS_URL = "redis://localhost:6379"
    DEFAULT_INDEX_NAME = "opea:index"
    DEFAULT_DOC_PREFIX = "doc:"

    def __init__(self, name: str, description: str = "", config: Dict = None):
        """Initialize RedisDBStore with configuration.

        Args:
            name: Store name
            description: Store description
            config: Configuration dictionary
        """
        super().__init__(name, description, config or {})

        # Configuration settings
        self.is_async = config.get("is_async", self.IS_ASYNC_DEFAULT)
        self.redis_url = config.get("REDIS_URL", self.DEFAULT_REDIS_URL)
        self.index_name = config.get("INDEX_NAME", self.DEFAULT_INDEX_NAME)
        self.doc_prefix = config.get("DOC_PREFIX", self.DEFAULT_DOC_PREFIX)
        self.auto_create_index = config.get("AUTO_CREATE_INDEX", True)

        # Connection state
        self.client = None
        self._async_initialized = False

        # Helper instances
        self._schema_generator = RedisSchemaGenerator()
        self._query_builder = RedisQueryBuilder()

    # ==========================================
    # Connection Management Methods
    # ==========================================

    def _initialize_connection_sync(self) -> bool:
        """Initialize the Redis connection synchronously for non-async mode."""
        if self.is_async:
            return True

        try:
            from redis import Redis

            self.client = Redis.from_url(self.redis_url, decode_responses=True)
            if not self.client.ping():
                raise ConnectionError(f"Failed to connect to Redis at {self.redis_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RedisDBStore: {e}")
            raise

    async def _ensure_async_initialized(self) -> None:
        """Ensure that async connection is initialized before using async methods."""
        if self.is_async and self.client is None and not self._async_initialized:
            await self._initialize_connection()
            self._async_initialized = True

    async def _initialize_connection(self) -> bool:
        """Initialize the Redis connection asynchronously."""
        try:
            from redis.asyncio import Redis

            self.client = Redis.from_url(self.redis_url, decode_responses=True)
            if not await self.client.ping():
                raise ConnectionError(f"Failed to connect to Redis at {self.redis_url}")
            return True
        except Exception:
            logger.exception("Failed to initialize RedisDBStore:")
            raise

    async def close(self) -> None:
        """Close the Redis connection."""
        try:
            if self.client:
                await self.client.close()
                logger.info("Closed Redis connection")
        except Exception:
            logger.exception("Error closing Redis connection:")

    def health_check(self) -> bool:
        """Perform a health check on the Redis connection.

        Returns:
            bool: True if the connection is healthy, False otherwise.
        """
        try:
            # For async mode, we can't perform health check during initialization
            # as the connection is initialized lazily. Return True to allow
            # the service to start, and let async methods handle connection issues.
            if self.is_async and not self._async_initialized:
                return True

            return self.client is not None
        except Exception:
            logger.exception("Health check failed:")
            return False

    def _build_index_command(self, sample_data: Dict) -> List[str]:
        """Build the Redis index creation command from sample data."""
        schema = self._schema_generator.generate_schema(sample_data)

        # Create JSON index using direct Redis command
        cmd = ["FT.CREATE", self.index_name, "ON", "JSON", "PREFIX", "1", self.doc_prefix, "SCHEMA"]

        # Add schema fields properly
        for field in schema:
            # Get the JSONPath and alias from the field
            json_path = field.name  # The JSONPath like "$.data.user"
            as_name = field.as_name  # The alias like "data_user"

            cmd.extend([json_path, "AS", as_name])

            field_type = str(type(field))
            if "TextField" in field_type:
                cmd.extend(["TEXT", "SORTABLE"])
            elif "NumericField" in field_type:
                cmd.extend(["NUMERIC", "SORTABLE"])
            elif "TagField" in field_type:
                cmd.extend(["TAG", "SORTABLE"])

        return cmd

    def _check_index_exists_sync(self) -> bool:
        """Check if index exists synchronously."""
        try:
            self.client.ft(self.index_name).info()
            return True
        except ResponseError:
            return False

    async def _check_index_exists(self) -> bool:
        """Check if index exists asynchronously."""
        try:
            await self.client.ft(self.index_name).info()
            return True
        except ResponseError:
            return False

    def create_index_sync(self, sample_data: Dict) -> None:
        """Create the Redis search index with flexible schema synchronously."""
        if self._check_index_exists_sync():
            return

        if not self.auto_create_index:
            raise RuntimeError(f"Index '{self.index_name}' does not exist and auto-creation is disabled")

        try:
            cmd = self._build_index_command(sample_data)
            result = self.client.execute_command(*cmd)
            logger.info(f"Created JSON index: {result}")
            logger.info(f"Created Redis index '{self.index_name}' with prefix '{self.doc_prefix}'")

            # Verify index creation
            self.client.ft(self.index_name).info()
        except Exception:
            logger.exception(f"Failed to create index '{self.index_name}':")
            raise

    async def create_index(self, sample_data: Dict) -> None:
        """Create the Redis search index with flexible schema asynchronously."""
        await self._ensure_async_initialized()

        if await self._check_index_exists():
            return

        if not self.auto_create_index:
            raise RuntimeError(f"Index '{self.index_name}' does not exist and auto-creation is disabled")

        try:
            cmd = self._build_index_command(sample_data)
            result = await self.client.execute_command(*cmd)
            logger.info(f"Created JSON index: {result}")
            logger.info(f"Created Redis index '{self.index_name}' with prefix '{self.doc_prefix}'")

            # Verify index creation
            await self.client.ft(self.index_name).info()
        except Exception:
            logger.exception(f"Failed to create index '{self.index_name}':")
            raise

    # ==========================================
    # Utility Methods
    # ==========================================

    def _generate_document_key(self) -> str:
        """Generate a unique key for document storage."""
        return f"{self.doc_prefix}{uuid.uuid4()}"

    def _ensure_document_id(self, doc: Dict, doc_id: str) -> None:
        """Ensure document has an ID field."""
        if not doc.get("id"):
            doc["id"] = doc_id

    def _process_search_result_document(self, doc, doc_id: str) -> Dict:
        """Process a single search result document."""
        if hasattr(doc, "json"):
            doc_dict = json.loads(doc.json)
        else:
            # Fallback: doc might be a dict already
            doc_dict = doc

        # Assign the Redis key as ID if the document doesn't have an ID field
        if not doc_dict.get("id"):
            doc_dict["id"] = doc_id

        return doc_dict

    # ==========================================
    # Document Storage Methods
    # ==========================================

    async def asave_document(self, doc: Dict, **kwargs) -> str:
        """Save a single document asynchronously.

        Args:
            doc: Document to save
            **kwargs: Additional arguments

        Returns:
            str: The key where the document was saved
        """
        await self._ensure_async_initialized()
        await self.create_index(doc)

        try:
            key = self._generate_document_key()
            await self.client.json().set(key, "$", doc)
            return key
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            raise

    async def asave_documents(self, docs: List[Dict], **kwargs) -> bool:
        """Save multiple documents asynchronously using pipeline for efficiency.

        Args:
            docs: List of documents to save
            **kwargs: Additional arguments

        Returns:
            bool: True if all documents were saved successfully
        """
        await self._ensure_async_initialized()

        if not docs:
            return True

        # Use first document for schema generation
        await self.create_index(docs[0])

        try:
            pipeline = self.client.pipeline()
            for doc in docs:
                key = self._generate_document_key()
                pipeline.json().set(key, "$", doc)

            results = await pipeline.execute()

            # Check for any failures
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Partial failure during batch save.")
                    return False

            return True
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
            return False

    async def aupdate_document(self, doc: Dict, **kwargs) -> bool:
        """Update a single document asynchronously.

        Args:
            doc: Document to update (must contain 'id' field)
            **kwargs: Additional arguments

        Returns:
            bool: True if document was updated successfully
        """
        await self._ensure_async_initialized()
        await self.create_index(doc)

        try:
            doc_id = doc.pop("id", None)
            if not doc_id:
                raise ValueError("Document must have an 'id' field for updates")

            result = await self.client.json().set(doc_id, "$", doc)
            return result is not None
        except Exception:
            logger.exception("Failed to update document:")
            raise

    async def aupdate_documents(self, docs: List[Dict], **kwargs) -> bool:
        """Update multiple documents asynchronously using pipeline for efficiency.

        Args:
            docs: List of documents to update
            **kwargs: Additional arguments

        Returns:
            bool: True if all documents were updated successfully
        """
        await self._ensure_async_initialized()

        if not docs:
            return True

        # Use first document for schema generation
        await self.create_index(docs[0])

        try:
            pipeline = self.client.pipeline()
            for doc in docs:
                # If doc has an id, use it with the prefix, otherwise generate one
                if "id" in doc:
                    key = f"{self.doc_prefix}{doc.pop('id')}"
                else:
                    key = self._generate_document_key()
                pipeline.json().set(key, "$", doc)

            results = await pipeline.execute()

            # Check for any failures
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Partial failure during batch update.")
                    return False

            return True
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            return False

    # ==========================================
    # Document Retrieval Methods
    # ==========================================

    async def aget_document_by_id(self, doc_id: str, **kwargs) -> Optional[Dict]:
        """Retrieve a single document by its ID.

        Args:
            doc_id: Document ID to retrieve
            **kwargs: Additional arguments

        Returns:
            Optional[Dict]: The document if found, None otherwise
        """
        await self._ensure_async_initialized()

        try:
            result = await self.client.json().get(doc_id)
            if result is not None:
                self._ensure_document_id(result, doc_id)
                return result
            return None
        except Exception:
            logger.exception(f"Failed to get document by ID {doc_id}:")
            return None

    async def aget_documents_by_ids(self, ids: List[str], **kwargs) -> List[Dict]:
        """Retrieve multiple documents by their IDs using pipeline for efficiency.

        Args:
            ids: List of document IDs to retrieve
            **kwargs: Additional arguments

        Returns:
            List[Dict]: List of found documents
        """
        await self._ensure_async_initialized()

        if not ids:
            return []

        try:
            pipeline = self.client.pipeline()
            for doc_id in ids:
                pipeline.json().get(doc_id)

            results = await pipeline.execute()

            processed_results = []
            for i, result in enumerate(results):
                if result:
                    doc = json.loads(result) if isinstance(result, str) else result
                    self._ensure_document_id(doc, ids[i])
                    processed_results.append(doc)

            return processed_results
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            raise

    # ==========================================
    # Document Deletion Methods
    # ==========================================

    async def adelete_document(self, doc_id: str, **kwargs) -> bool:
        """Delete a single document by its ID.

        Args:
            doc_id: Document ID to delete
            **kwargs: Additional arguments

        Returns:
            bool: True if document was deleted successfully
        """
        await self._ensure_async_initialized()

        try:
            deleted = await self.client.delete(doc_id)
            return deleted > 0
        except Exception as e:
            logger.exception(f"Failed to delete document {doc_id}:")
            raise

    async def adelete_documents(self, ids: List[str], **kwargs) -> bool:
        """Delete multiple documents by their IDs.

        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments

        Returns:
            bool: True if any documents were deleted
        """
        await self._ensure_async_initialized()

        if not ids:
            return True

        try:
            keys = [f"{self.doc_prefix}{doc_id}" for doc_id in ids]
            deleted = await self.client.delete(*keys)
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise

    # ==========================================
    # Search Methods
    # ==========================================

    async def asearch(self, key: str, value: Any, search_type: str = "exact", **kwargs) -> List[Dict]:
        """Search for documents by key and value using Redis Search.

        Args:
            key: Field name to search on
            value: Value to search for
            search_type: Type of search ("exact", "contains", etc.)
            **kwargs: Additional arguments

        Returns:
            List[Dict]: List of matching documents
        """
        await self._ensure_async_initialized()

        try:
            query_str = self._query_builder.build_search_query(key, value, search_type)
            search_result = await self.client.ft(self.index_name).search(query_str)

            # Convert results to list of dictionaries
            results = []
            for doc in search_result.docs:
                doc_dict = self._process_search_result_document(doc, doc.id)
                results.append(doc_dict)

            return results
        except Exception as e:
            logger.error(f"Search failed for key '{key}' with value '{value}': {e}")
            raise

    async def asearch_by_keyword(self, keyword: str, max_results: int = 5, **kwargs) -> list[dict]:
        """Search for documents by keyword across all text fields with BM25 relevance scoring.

        Args:
            keyword: Keyword to search for
            max_results: Maximum number of results to return
            **kwargs: Additional arguments

        Returns:
            List[Dict]: List of matching documents with BM25 relevance scores
        """
        await self._ensure_async_initialized()

        try:
            # Escape special characters in the keyword
            escaped_keyword = RedisQueryBuilder.escape_query_value(keyword)

            # Create a query that searches across all text fields for the keyword
            # Use a more general query syntax that searches within text fields
            query_str = f"*{escaped_keyword}*"

            # Use Query object to properly set limit and enable scoring
            query = Query(query_str).paging(0, max_results).with_scores()
            search_result = await self.client.ft(self.index_name).search(query)

            # Convert results to list of dictionaries with BM25 scores
            results = []
            for doc in search_result.docs:
                doc_dict = self._process_search_result_document(doc, doc.id)
                # Add BM25 relevance score to the document
                doc_dict["score"] = float(doc.score) if hasattr(doc, "score") else 0.0
                results.append(doc_dict)

            # Sort results by BM25 score in descending order (highest scores first)
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            return results
        except Exception as e:
            logger.exception(f"Keyword search failed for '{keyword}':")
            raise
