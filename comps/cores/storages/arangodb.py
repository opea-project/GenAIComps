# Copyright (C) 2025 ArangoDB Inc.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from ..common.storage import OpeaStore
from ..mega.logger import CustomLogger

logger = CustomLogger("ArangoDBStore")


class ArangoDBStore(OpeaStore):
    """A concrete implementation of OpeaStore for ArangoDB."""

    IS_ASYNC_DEFAULT = True

    def __init__(self, name: str, description: str = "", config: dict = {}):
        """Initializes the ArangoDBStore with the given configuration.

        Args:
            name (str): The name of the component.
            description (str): A brief description of the component.
            config (dict, optional): Configuration parameters for the component, namely:
                - ARANGODB_HOST: The host URL for the ArangoDB instance.
                - ARANGODB_USERNAME: The username for authentication.
                - ARANGODB_PASSWORD: The password for authentication.
                - ARANGODB_DB_NAME: The name of the database to connect to.
                - ARANGODB_COLLECTION_NAME: The name of the collection to use.
                - client: An instance of arango.ArangoClient (optional).
                - db: An instance of arango.database.StandardDatabase (optional).
                - collection: An instance of arango.collection.StandardCollection (optional).
        """

        is_async = config.get("is_async", self.IS_ASYNC_DEFAULT)

        if is_async:
            try:
                from arangoasync import ArangoClient
                from arangoasync.collection import StandardCollection
                from arangoasync.database import StandardDatabase
            except ImportError:
                m = "ArangoDB async client library is not installed. Please install it using 'pip install python-arango-async'."
                logger.error(m)
                raise
        else:
            try:
                from arango import ArangoClient
                from arango.collection import StandardCollection
                from arango.database import StandardDatabase
            except ImportError:
                m = "ArangoDB client library is not installed. Please install it using 'pip install python-arango'."
                logger.error(m)
                raise

        super().__init__(name, description, config)

        self.client: ArangoClient = config.get("client", None)
        self.db: StandardDatabase = config.get("db", None)
        self.collection: StandardCollection = config.get("collection", None)
        self.is_async = is_async
        self._async_initialized = False

        # self.async_db = self.db.begin_async_execution()
        # self.async_collection = self.async_db.collection(self.collection.name)

    def _initialize_connection_sync(self) -> None:
        """Initializes the connection to the ArangoDB database and collection synchronously."""
        if not self.is_async:
            self._initialize_connection()

    async def _initialize_connection_async_wrapper(self) -> None:
        """Initializes the connection to the ArangoDB database and collection asynchronously."""
        if self.is_async:
            await self._initialize_connection_async()

    async def _ensure_async_initialized(self):
        """Ensures that async connection is initialized before using async methods."""
        if self.is_async and not self._async_initialized:
            await self._initialize_connection_async()
            self._async_initialized = True

    def _initialize_connection(self) -> None:
        """Initializes the connection to the ArangoDB database and collection."""

        from arango import ArangoClient

        try:
            host = self.config.get("ARANGODB_HOST", "http://localhost:8529")
            username = self.config.get("ARANGODB_USERNAME", "root")
            password = self.config.get("ARANGODB_PASSWORD", "")
            database_name = self.config.get("ARANGODB_DB_NAME", "_system")
            collection_name = self.config.get("ARANGODB_COLLECTION_NAME", "default")

            if not self.client:
                self.client = ArangoClient(hosts=host)

            if not self.db:
                self.db = self.client.db(database_name, username=username, password=password, verify=True)

            if not self.collection:
                if not self.db.has_collection(collection_name):
                    self.collection = self.db.create_collection(collection_name)
                else:
                    self.collection = self.db.collection(collection_name)

            logger.info(f"Connected to ArangoDB database '{database_name}' and collection '{collection_name}'.")
        except Exception as e:
            logger.exception("Failed to initialize ArangoDB connection:")
            raise

    async def _initialize_connection_async(self) -> None:
        """Initializes the connection to the ArangoDB database and collection asynchronously."""

        from arangoasync import ArangoClient
        from arangoasync.auth import Auth

        try:
            host = self.config.get("ARANGODB_HOST", "http://localhost:8529")
            username = self.config.get("ARANGODB_USERNAME", "root")
            password = self.config.get("ARANGODB_PASSWORD", "")
            database_name = self.config.get("ARANGODB_DB_NAME", "_system")
            collection_name = self.config.get("ARANGODB_COLLECTION_NAME", "default")

            if not self.client:
                self.client = ArangoClient(hosts=host)

            auth = Auth(username=username, password=password)

            if not self.db:
                self.db = await self.client.db(database_name, auth=auth)

            if not self.collection:
                if await self.db.has_collection(collection_name):
                    self.collection = self.db.collection(collection_name)
                else:
                    self.collection = await self.db.create_collection(collection_name)

            logger.info(
                f"Connected to ArangoDB database '{database_name}' and collection '{collection_name}' asynchronously."
            )
        except Exception as e:
            logger.exception("Failed to initialize ArangoDB async connection:")
            raise

    def health_check(self) -> bool:
        """Performs a health check on the ArangoDB connection.

        Returns:
            bool: True if the connection is healthy, False otherwise.
        """
        try:
            # For async mode, we can't perform health check during initialization
            # as the connection is initialized lazily. Return True to allow
            # the service to start, and let async methods handle connection issues.
            if self.is_async and not self._async_initialized:
                return True

            if self.db is None:
                return False

            self.db.version()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def save_document(self, doc: dict, **kwargs) -> bool | dict:
        """Save a single document to the store.
        Document can optionally contain a unique identifier.

        Args:
            doc (dict): The document data to save.
            **kwargs: Additional arguments for saving the document.

        Returns:
            bool | dict: The result of the save operation.
        """
        try:
            return self.collection.insert(doc, **kwargs)
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            raise

    async def asave_document(self, doc: dict, **kwargs) -> bool | dict:
        """Save a single document to the store.
        Document can optionally contain a unique identifier.

        Args:
            doc (dict): The document data to save.
            **kwargs: Additional arguments for saving the document.

        Returns:
            bool | dict: The result of the save operation.
        """
        await self._ensure_async_initialized()
        try:
            doc.pop("_id", None)
            metadata = await self.collection.insert(doc)
            doc_id = metadata["_id"]

            return doc_id
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            logger.exception("Failed to save document.")
            raise

    async def aupdate_document(self, doc: dict, **kwargs) -> bool | dict:
        """Update a single document in the store asynchronously.
        Document must contain its unique identifier.

        Args:
            doc (dict): The document data to update.
            **kwargs: Additional arguments for updating the document.

        Returns:
            bool | dict: The result of the update operation.
        """
        await self._ensure_async_initialized()
        try:
            metadata = await self.collection.update(doc, **kwargs)
            doc_id = metadata.get("_id", None)
            if doc_id:
                logger.debug(f"Updated document: {doc_id}")
                return True
            else:
                return False
        except Exception as e:
            logger.exception("Failed to update document by id {id}:")
            raise

    async def aget_document_by_id(self, id: str, **kwargs) -> dict | None:
        """Retrieve a single document by its unique identifier asynchronously.

        Args:
            id (str): The unique identifier for the document.
            **kwargs: Additional arguments for retrieving the document.

        Returns:
            dict: The retrieved document data.
        """
        await self._ensure_async_initialized()
        try:
            return await self.collection.get(id, **kwargs)
        except Exception as e:
            logger.exception(f"Failed to retrieve document by ID {id}:")
            raise

    async def adelete_document(self, id: str, **kwargs) -> bool | dict:
        """Delete a single document from the store asynchronously.

        Args:
            id (str): The unique identifier for the document.
            **kwargs: Additional arguments for deleting the document.

        Returns:
            bool | dict: The result of the delete operation.
        """
        await self._ensure_async_initialized()
        try:
            metadata = await self.collection.delete(id, **kwargs)
            doc_id = metadata.get("_id", None)
            if doc_id:
                logger.debug(f"Deleted document: {doc_id}")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Failed to delete document by ID {id}: {e}")
            logger.exception("Failed to delete document.")
            raise

    async def asearch(self, key: str, value: Any, search_type: str = "exact", **kwargs) -> list[dict]:
        """Search for documents in the store based on a specific key-value pair asynchronously.

        Args:
            key (str): The key to search for.
            value (str): The value to search for.
            search_type (str): The type of search to perform. Options include:
                - "exact": Exact match.
                - "contains": Contains the value.
                - "starts_with": Starts with the value.
                - "ends_with": Ends with the value.
                - "regex": Regular expression match.
                - "custom": Custom filter clause. In this case,
                    the `filter_clause` argument must be provided as a string
                    in `kwargs`.
            **kwargs: Additional arguments passed to the query execution.

        Returns:
            list[dict]: A list of documents matching the search criteria.
        """
        await self._ensure_async_initialized()

        # Handle nested field access (e.g., "data.user" -> doc.data.user)
        if "." in key:
            field_access = "doc." + key
        else:
            field_access = "doc[@key]"
            # Only use bind parameter for single fields
            use_bind_key = True

        # For nested fields, we don't use bind parameters for the key
        if "." in key:
            use_bind_key = False
            if search_type == "exact":
                filter_clause = f"{field_access} == @value"
            elif search_type == "contains":
                filter_clause = f"CONTAINS({field_access}, @value)"
            elif search_type == "starts_with":
                filter_clause = f"STARTS_WITH({field_access}, @value)"
            elif search_type == "ends_with":
                filter_clause = f"ENDS_WITH({field_access}, @value)"
            elif search_type == "regex":
                filter_clause = f"REGEX_MATCHES({field_access}, @value)"
        else:
            # Use the existing bind parameter approach for single fields
            if search_type == "exact":
                filter_clause = "doc[@key] == @value"
            elif search_type == "contains":
                filter_clause = "CONTAINS(doc[@key], @value)"
            elif search_type == "starts_with":
                filter_clause = "STARTS_WITH(doc[@key], @value)"
            elif search_type == "ends_with":
                filter_clause = "ENDS_WITH(doc[@key], @value)"
            elif search_type == "regex":
                filter_clause = "REGEX_MATCHES(doc[@key], @value)"

        if search_type == "custom":
            filter_clause = kwargs.pop("filter_clause")
            if not filter_clause or not isinstance(filter_clause, str):
                raise ValueError("Custom filter clause is a required string for 'custom' search type.")
        elif search_type not in ["exact", "contains", "starts_with", "ends_with", "regex"]:
            raise ValueError(f"Unsupported search type: {search_type}")

        query = f"""
            FOR doc IN @@col
                FILTER {filter_clause}
                RETURN doc
        """

        try:
            # Set up bind vars based on whether we're using bind parameters for the key
            if "." in key:
                bind_vars = {"@col": self.collection.name, "value": value}
            else:
                bind_vars = {"@col": self.collection.name, "key": key, "value": value}
            cursor = await self.db.aql.execute(query, bind_vars=bind_vars, **kwargs)
            result = [doc async for doc in cursor]
            return result
        except Exception as e:
            logger.error(f"Failed to search documents with {key} / {value}: {e}. Query: {query}")
            logger.exception("Failed to search documents.")
            raise

    async def asearch_by_keyword(self, keyword: str, max_results: int = 5, **kwargs) -> list[dict]:
        """Asynchronously search for documents based on a keyword using full text search.

        Args:
            keyword (str): The keyword to search for.
            max_results (int): The maximum number of results to return.
            **kwargs: Additional arguments for the search.

        Returns:
            list[dict]: A list of matching documents sorted by relevance.
        """
        await self._ensure_async_initialized()

        try:
            # Create a full-text search view if it doesn't exist
            view_name = f"{self.collection.name}_search"

            # Check if a collection with the same name exists (which would conflict)
            if await self.db.has_collection(view_name):
                raise Exception(f"A collection named '{view_name}' exists, cannot create view with same name.")

            # Check if view already exists and delete it to recreate with fresh settings
            try:
                existing_view = await self.db.view(view_name)
                if existing_view:
                    await self.db.delete_view(view_name)
                    logger.info(f"Deleted existing view: {view_name}")
            except Exception as e:
                # View doesn't exist, which is fine
                logger.debug(f"View '{view_name}' doesn't exist yet: {e}")

            # Create the view
            try:
                view_properties = {
                    "links": {self.collection.name: {"includeAllFields": True, "analyzers": ["text_en", "identity"]}}
                }
                view = await self.db.create_view(name=view_name, view_type="arangosearch", properties=view_properties)
                logger.info(f"Created view: {view_name}")
            except Exception:
                logger.exception("Failed to create search view:")
                raise

            # Build the filter clause based on specified fields
            if "fields" in kwargs and isinstance(kwargs["fields"], list) and kwargs["fields"]:
                fields = kwargs.pop("fields")
                field_clauses = [f'PHRASE(doc.{field}, @keyword, "text_en")' for field in fields]
                filter_clause = " OR ".join(field_clauses)
            else:
                raise ValueError("Fields parameter is required and must be a non-empty list for keyword search.")

            # Perform full-text search using the view
            query = f"""
                FOR doc IN {view_name}
                    SEARCH {filter_clause}
                    SORT BM25(doc) DESC
                    LIMIT @max_results
                    RETURN MERGE(doc, {{ "score": BM25(doc) }})
            """

            bind_vars = {"keyword": keyword, "max_results": max_results}

            cursor = await self.db.aql.execute(query, bind_vars=bind_vars)
            result = [doc async for doc in cursor]

            if result:
                logger.info(f"Found {len(result)} results")
                return result
            else:
                logger.info("No results from ArangoSearch")
                return []

        except Exception:
            logger.exception(f"Failed to search by keyword {keyword}:")
            raise

    def save_documents(self, docs: list[dict], **kwargs) -> bool | list:
        """Save multiple documents to the store.
        Documents can optionally contain unique identifiers.

        NOTE: If inserting a document fails, the exception is not raised
        but returned as an object in the result list. It is up to you to
        inspect the list to determine which documents were
        inserted successfully (returns document metadata)
        and which were not (returns exception object).
        Alternatively, you can rely on setting
        raise_on_document_error to True (defaults to False).

        Args:
            docs (list[dict]): A list of document data to save.
            **kwargs: Additional arguments for saving the documents.

        Returns:
            bool | list: A list of results for the save operations.
        """
        try:
            return self.collection.insert_many(docs, **kwargs)
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
            raise

    def update_document(self, doc: dict, **kwargs) -> bool | dict:
        """Update a single document in the store.
        Document must contain its unique identifier.

        Args:
            doc (dict): The document data to update.
            **kwargs: Additional arguments for updating the document.

        Returns:
            bool | dict: The result of the update operation.
        """
        try:
            return self.collection.update(doc, **kwargs)
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            raise

    def update_documents(self, docs: list[dict], **kwargs) -> bool | list:
        """Update multiple documents in the store.
        Each document must contain its unique identifier.

        NOTE: If updating a document fails, the exception is not raised
        but returned as an object in the result list. It is up to you to
        inspect the list to determine which documents were
        updated successfully (returns document metadata)
        and which were not (returns exception object).
        Alternatively, you can rely on setting
        raise_on_document_error to True (defaults to False).

        Args:
            docs (list[dict]): The list of documents to update.
            **kwargs: Additional arguments for updating the documents.

        Returns:
            bool | list: A list of results for the update operations.
        """
        try:
            return self.collection.update_many(docs, **kwargs)
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            raise

    def get_document_by_id(self, id: str, **kwargs) -> dict | None:
        """Retrieve a single document by its unique identifier.

        Args:
            id (str): The unique identifier for the document.
            **kwargs: Additional arguments for retrieving the document.

        Returns:
            dict: The retrieved document data.
        """
        try:
            return self.collection.get(id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to retrieve document by ID {id}: {e}")
            raise

    def get_documents_by_ids(self, ids: list[str], **kwargs) -> list[dict]:
        """Retrieve multiple documents by their unique identifiers.

        Args:
            ids (list[str]): A list of unique identifiers for the documents.
            **kwargs: Additional arguments for retrieving the documents.

        Returns:
            list[dict]: A list of retrieved document data.
        """
        try:
            return self.collection.get_many(ids, **kwargs)
        except Exception as e:
            logger.error(f"Failed to retrieve documents by IDs {ids}: {e}")
            raise

    def delete_document(self, id: str, **kwargs) -> bool | dict:
        """Delete a single document from the store.

        Args:
            id (str): The unique identifier for the document.
            **kwargs: Additional arguments for deleting the document.

        Returns:
            bool | dict: The result of the delete operation.
        """
        try:
            return self.collection.delete(id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete document by ID {id}: {e}")
            raise

    def delete_documents(self, ids: list[str], **kwargs) -> bool | list:
        """Delete multiple documents from the store.

        NOTE: If updating a document fails, the exception is not raised
        but returned as an object in the result list. It is up to you to
        inspect the list to determine which documents were
        updated successfully (returns document metadata)
        and which were not (returns exception object).
        Alternatively, you can rely on setting
        raise_on_document_error to True (defaults to False).

        Args:
            ids (list[str]): A list of unique identifiers for the documents.
        **kwargs: Additional arguments for deleting the documents.

        Returns:
            bool | list: A list of results for the delete operations.
        """
        try:
            return self.collection.delete_many(ids, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete documents by IDs {ids}: {e}")
            raise

    def search(self, key: str, value: Any, search_type: str = "exact", **kwargs) -> list[dict]:
        """Search for documents in the store based on a specific key-value pair.

        Args:
            key (str): The key to search for.
            value (str): The value to search for.
            search_type (str): The type of search to perform. Options include:
                - "exact": Exact match.
                - "contains": Contains the value.
                - "starts_with": Starts with the value.
                - "ends_with": Ends with the value.
                - "regex": Regular expression match.
                - "custom": Custom filter clause. In this case,
                    the `filter_clause` argument must be provided as a string
                    in `kwargs`.
            **kwargs: Additional arguments passed to the query execution.

        Returns:
            list[dict]: A list of documents matching the search criteria.
        """

        # Handle nested field access (e.g., "data.user" -> doc.data.user)
        if "." in key:
            field_access = "doc." + key
        else:
            field_access = "doc[@key]"
            # Only use bind parameter for single fields
            use_bind_key = True

        # For nested fields, we don't use bind parameters for the key
        if "." in key:
            use_bind_key = False
            if search_type == "exact":
                filter_clause = f"{field_access} == @value"
            elif search_type == "contains":
                filter_clause = f"CONTAINS({field_access}, @value)"
            elif search_type == "starts_with":
                filter_clause = f"STARTS_WITH({field_access}, @value)"
            elif search_type == "ends_with":
                filter_clause = f"ENDS_WITH({field_access}, @value)"
            elif search_type == "regex":
                filter_clause = f"REGEX_MATCHES({field_access}, @value)"
        else:
            # Use the existing bind parameter approach for single fields
            if search_type == "exact":
                filter_clause = "doc[@key] == @value"
            elif search_type == "contains":
                filter_clause = "CONTAINS(doc[@key], @value)"
            elif search_type == "starts_with":
                filter_clause = "STARTS_WITH(doc[@key], @value)"
            elif search_type == "ends_with":
                filter_clause = "ENDS_WITH(doc[@key], @value)"
            elif search_type == "regex":
                filter_clause = "REGEX_MATCHES(doc[@key], @value)"

        if search_type == "custom":
            filter_clause = kwargs.pop("filter_clause")
            if not filter_clause or not isinstance(filter_clause, str):
                raise ValueError("Custom filter clause is a required string for 'custom' search type.")
        elif search_type not in ["exact", "contains", "starts_with", "ends_with", "regex"]:
            raise ValueError(f"Unsupported search type: {search_type}")

        query = f"""
            FOR doc IN @@col
                FILTER {filter_clause}
                RETURN doc
        """

        try:
            # Set up bind vars based on whether we're using bind parameters for the key
            if "." in key:
                bind_vars = {"@col": self.collection.name, "value": value}
            else:
                bind_vars = {"@col": self.collection.name, "key": key, "value": value}
            return list(self.db.aql.execute(query, bind_vars=bind_vars, **kwargs))
        except Exception as e:
            logger.error(f"Failed to search documents with {key} / {value}: {e}. Query: {query}")
            raise
