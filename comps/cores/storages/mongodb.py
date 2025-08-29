# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import bson.errors as BsonError
import motor.motor_asyncio as motor
from bson.objectid import ObjectId

from ..common.storage import OpeaStore
from ..mega.logger import CustomLogger

logger = CustomLogger("MongoDBStore")


class MongoDBStore(OpeaStore):

    def __init__(self, name: str, description: str = "", config: dict = {}):
        super().__init__(name, description, config)
        self.user = config.get("user", None)

    def _initialize_db(self) -> None:
        """Initializes the MongoDB database connection and collection."""

        MONGO_HOST = self.config.get("MONGO_HOST", "localhost")
        MONGO_PORT = self.config.get("MONGO_PORT", 27017)
        DB_NAME = self.config.get("DB_NAME", "OPEA")
        COLLECTION_NAME = self.config.get("COLLECTION_NAME", "default")
        conn_url = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/"

        try:
            client = motor.AsyncIOMotorClient(conn_url)
            self.db = client[DB_NAME]

        except Exception as e:
            logger.error(e)
            raise Exception()

        self.collection = self.db[COLLECTION_NAME]

    def health_check(self) -> bool:
        """Performs a health check on the MongoDB connection.

        Returns:
            bool: True if the connection is healthy, False otherwise.
        """
        try:
            self.collection.count_documents({}, limit=1)
            logger.info("MongoDB Health check succeed!")
            return True
        except Exception as e:
            logger.error(f"MongoDB Health check failed: {e}")
            return False

    async def asave_document(self, doc: dict, **kwargs) -> bool | dict:
        """Stores a new document into the MongoDB collection.

        Args:
            doc (dict): The document data to save.
            **kwargs: Additional arguments for saving the document.

        Returns:
            bool | dict: The result of the save operation.
        """
        try:
            if "doc_id" in doc:
                del doc["doc_id"]

            inserted_data = await self.collection.insert_one(doc)
            doc_id = str(inserted_data.inserted_id)
            logger.info(f"Inserted document: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Fail to save document: {e}")
            raise Exception(e)

    async def asave_documents(self, docs: list[dict], **kwargs) -> bool | list:
        """Save multiple documents to the store.

        Args:
            docs (list[dict]): A list of document data to save.
            **kwargs: Additional arguments for saving the documents.

        Returns:
            bool | list: A list of results for the save operations.
        """
        try:
            inserted_data = await self.collection.insert_many(
               [{key: value for key, value in doc.items() if key != "doc_id"} for doc in docs]
            )
            doc_ids = str(inserted_data.inserted_ids)
            logger.info(f"Inserted documents: {doc_ids}")
            return doc_ids

        except Exception as e:
            logger.error(f"Fail to save document: {e}")
            raise Exception(e)

    async def aupdate_document(self, doc: dict, **kwargs) -> bool | dict:
        """Update a single document in the store.

        Args:
            doc (dict): The document data to update.
            **kwargs: Additional arguments for updating the document.

        Returns:
            bool | dict: The result of the update operation.
        """
        try:
            doc_id = doc.get("doc_id", None)
            _id = ObjectId(doc_id)
            first_query = doc.get("first_query", None)
            data_dict = doc.get("data", None)
            if first_query:
                data = {"data": data_dict, "first_query": first_query}
            else:
                data = {"data": data_dict}

            updated_result = await self.collection.update_one(
                {"_id": _id, "user": self.user},
                {"$set": data},
            )

            if updated_result.modified_count == 1:
                logger.info(f"Updated document: {doc_id}")
                return True
            else:
                raise Exception("Not able to update the data.")

        except BsonError.InvalidId as e:
            logger.error(e)
            raise KeyError(e)

        except Exception as e:
            logger.error(e)
            raise Exception(e)

    async def aupdate_documents(self, docs: list[dict], **kwargs) -> bool | dict:
        """Update multiple documents in the store.

        Args:
            docs (list[dict]): The list of documents to update.
            **kwargs: Additional arguments for updating the documents.

        Returns:
            bool | dict: The result of the update operation.
        """
        for doc in docs:
            result = await self.aupdate_document(doc)
            assert result
        return True

    async def aget_document_by_id(self, id: str, **kwargs) -> dict | None:
        """Asynchronously retrieve a single document by its unique identifier.

        Args:
            id (str): The unique identifier for the document.
            **kwargs: Additional arguments for retrieving the document.

        Returns:
            dict | None: The user's feedback data if found, None otherwise.

        Raises:
            Exception: If there is an error while retrieving data.
        """
        try:
            _id = ObjectId(id)
            response: dict | None = await self.collection.find_one({"_id": _id, "user": self.user})
            if response:
                del response["_id"]
                logger.info(f"Retrieved document: {id}")
                return response
            return None

        except BsonError.InvalidId as e:
            logger.info(e)
            raise KeyError(e)

        except Exception as e:
            logger.info(e)
            raise Exception(e)

    async def aget_documents_by_ids(self, ids: list[str], **kwargs) -> list[dict]:
        """Asynchronously retrieve a single document by its unique identifier.

        Args:
            id (str): The unique identifier for the document.
            **kwargs: Additional arguments for retrieving the document.

        Returns:
            dict: The retrieved document data.
        """
        try:
            responses = []
            for id in ids:
                _id = ObjectId(id)
                response: dict | None = await self.collection.find_one({"_id": _id, "user": self.user})
                if response:
                    del response["_id"]
                    responses.append(response)
            logger.info(f"Retrieved documents: {response}")
            return responses

        except BsonError.InvalidId as e:
            logger.info(e)
            raise KeyError(e)

        except Exception as e:
            logger.info(e)
            raise Exception(e)

    async def aget_documents_by_user(self, user: str = None, **kwargs) -> list[dict] | None:
        """Asynchronously retrieve all documents for a specific user.

        Args:
            user (str): The unique identifier for the user.
            **kwargs: Additional arguments for retrieving the documents.

        Returns:
            list[dict] | None: List of dict of feedback data of the user, None otherwise.

        Raises:
            Exception: If there is an error while retrieving data.
        """
        try:
            responses = []
            if user is None:
                user = self.user
                
            async for document in self.collection.find({"user": user}, {"data": 0}):
                document["doc_id"] = str(document["_id"])
                del document["_id"]
                responses.append(document)
            logger.info(f"Retrieved documents: {responses}")
            return responses

        except Exception as e:
            logger.info(e)
            raise Exception(e)

    async def adelete_document(self, id: str, **kwargs) -> bool:
        """Asynchronously delete a single document from the store.

        Args:
            id (str): The unique identifier for the document.
            **kwargs: Additional arguments for deleting the document.

        Returns:
            bool: True if doc is successfully deleted, False otherwise.

        Raises:
            KeyError: If the provided id is invalid:
            Exception: If any errors occurs during delete process.
        """
        try:
            _id = ObjectId(id)
            result = await self.collection.delete_one({"_id": _id, "user": self.user})

            delete_count = result.deleted_count
            logger.info(f"Deleted {delete_count} documents!")

            return True if delete_count == 1 else False

        except BsonError.InvalidId as e:
            logger.error(e)
            raise KeyError(e)

        except Exception as e:
            logger.error(e)
            raise Exception(e)

    async def adelete_documents(self, ids: list[str], **kwargs) -> bool:
        """Asynchronously delete multiple documents from the store.".

        Args:
            ids (list[str]): A list of unique identifiers for the documents.
            **kwargs: Additional arguments for deleting the documents.

        Returns:
            bool: True if doc is successfully deleted, False otherwise.

        Raises:
            KeyError: If the provided id is invalid:
            Exception: If any errors occurs during delete process.
        """
        try:
            result = await self.collection.delete_many({"_id": {"$in": ids}, "user": self.user})

            delete_count = result.deleted_count
            logger.info(f"Deleted {delete_count} documents!")

            return True if delete_count == 1 else False

        except BsonError.InvalidId as e:
            logger.error(e)
            raise KeyError(e)

        except Exception as e:
            logger.error(e)
            raise Exception(e)

    async def asearch(self, key: str, value: Any = None, search_type: str = "exact", **kwargs) -> list[dict]:
        """Asynchronously search for documents based on a key-value pair.

        Args:
            key (str): The keyword of prompt to search for.
            value (Any): The value to match against the key.
            search_type (str): The type of search to perform.
            **kwargs: Additional arguments for the search.

        Returns:
            list[dict]: A list of matching documents.
        """
        try:
            # Create a text index if not already created
            self.collection.create_index([("$**", "text")])
            # Perform text search
            # results = await self.collection.find({"$text": {"$search": key}}, {"score": {"$meta": "textScore"}})
            # sorted_results = results.sort([("score", {"$meta": "textScore"})])

            # # Return a list of top 5 most relevant data
            # relevant_data = await sorted_results.to_list(length=5)

            relevant_data = await self.collection.find({"$text": {"$search": key}}, {"score": {"$meta": "textScore"}}).sort([("score", {"$meta": "textScore"})]).to_list(length=5)

            # Serialize data and return
            serialized_data = [
                {"id": str(doc["_id"]), "data": doc["prompt_text"], "user": doc["user"], "score": doc["score"]}
                for doc in relevant_data
            ]

            logger.info(f"Search results: {serialized_data}")
            return serialized_data

        except Exception as e:
            logger.info(e)
            raise Exception(e)
