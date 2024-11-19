# # Copyright (C) 2024 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from config import COLLECTION_NAME
from arango_conn import ArangoClient
from pydantic import BaseModel

class DocumentStore:

    def __init__(
        self,
        user: str,
    ):
        self.user = user

    def initialize_storage(self) -> None:
        try:
            self.db_client = ArangoClient.get_db_client()
            # Create collection if it doesn't exist
            print(COLLECTION_NAME)
            if not self.db_client.has_collection(COLLECTION_NAME):
                print("Creating collection")
                self.collection = self.db_client.create_collection(COLLECTION_NAME)
            else:
                print("Collection already exists")
                self.collection = self.db_client.collection(COLLECTION_NAME)
            
            print(f"Successfully initialized storage with collection: {COLLECTION_NAME}")
            
        except Exception as e:
            print(f"Failed to initialize storage: {e}, url: {ArangoClient.conn_url}, collection: {COLLECTION_NAME}")
            raise Exception(f"Storage initialization failed: {e}, url: {ArangoClient.conn_url}, collection: {COLLECTION_NAME}")

    def save_document(self, document: BaseModel) -> str:
        """Stores a new document into the storage.

        Args:
            document: The document to be stored. It should be a Pydantic model.

        Returns:
            str: The ID of the inserted document.

        Raises:
            Exception: If an error occurs while storing the document.
        """
        try:
            inserted_conv = self.collection.insert(
                document.model_dump(by_alias=True, mode="json", exclude={"id"})
            )
            document_id = str(inserted_conv["_key"])
            return document_id

        except Exception as e:
            print(e)
            raise Exception(e)
            
    def update_document(self, document_id, updated_data, first_query) -> str:
        """Updates a document in the collection with the given document_id.

        Args:
            document_id (str): The ID of the document to update.
            updated_data (object): The updated data to be set in the document.
            first_query (object): The first query to be set in the document.

        Returns:
            bool: True if the document was successfully updated, False otherwise.

        Raises:
            KeyError: If an invalid document_id is provided.
            Exception: If an error occurs during the update process.
        """
        try:
            cursor = self.db_client.aql.execute(f"""
                FOR doc IN @@collection
                    FILTER doc._key == @document_id AND doc.data.user == @user
                    LIMIT 1
                    UPDATE doc WITH @body IN @@collection
                    OPTIONS {{ keepNull: @keep_none, mergeObjects: @merge }}
            """,
                bind_vars={"@collection": self.collection.name, "document_id": document_id, "user": self.user, "body": {"data": updated_data.model_dump(by_alias=True, mode="json"), "first_query": first_query}, "keep_none": True, "merge": True}
            )
            return "Updated document : {}".format(document_id)

        except Exception as e:
            print(e)
            raise Exception(e)

    def get_all_documents_of_user(self) -> list[dict]:
        """Retrieves all documents of a specific user from the collection.

        Returns:
            A list of dictionaries representing the conversation documents.
        Raises:
            Exception: If there is an error while retrieving the documents.
        """
        conversation_list: list = []
        try:
            cursor = self.db_client.aql.execute("""
                FOR doc IN @@collection
                    FILTER doc.data.user == @user
                    RETURN doc
            """,
                bind_vars={"@collection": self.collection.name, "user": self.user}
            )
            for document in cursor:
                document["id"] = document["_key"]
                del document["_key"]
                del document["_id"]
                del document["_rev"]
                del document["data"]
                conversation_list.append(document)
            return conversation_list

        except Exception as e:
            print(e)
            raise Exception(e)

    def get_user_documents_by_id(self, document_id) -> dict | None:
        """Retrieves a user document from the collection based on the given document ID.

        Args:
            document_id (str): The ID of the document to retrieve.

        Returns:
            dict | None: The user document if found, None otherwise.
        """
        try:
            response = self.collection.get(document_id)
            if response and response['data']["user"] == self.user:
                response.pop("_id", None)
                return response
            return None

        except Exception as e:
            print(e)
            raise Exception(e)

    def delete_document(self, document_id) -> str:
        """Deletes a document from the collection based on the provided document ID.

        Args:
            document_id (str): The ID of the document to be deleted.

        Returns:
            bool: True if the document is successfully deleted, False otherwise.

        Raises:
            KeyError: If the provided document ID is invalid.
            Exception: If an error occurs during the deletion process.
        """
        try:
            doc = self.collection.get(document_id)
            if doc and doc['data']["user"] == self.user:
                self.collection.delete(document_id)
                return "Deleted document : {}".format(document_id)
            else:
                raise Exception("Not able to delete the Document")

        except Exception as e:
            print(e)
            raise Exception(e)
