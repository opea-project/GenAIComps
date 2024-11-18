# # Copyright (C) 2024 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0


import bson.errors as BsonError
from bson.objectid import ObjectId
from config import COLLECTION_NAME
from arango_conn import ArangoClient
from pydantic import BaseModel
import asyncio

class TestDocument(BaseModel):
    messages: str
    user: str

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
            if not self.db_client.has_collection(COLLECTION_NAME):
                self.collection = self.db_client.create_collection(COLLECTION_NAME)
            else:
                self.collection = self.db_client.collection(COLLECTION_NAME)
            
            print(f"Successfully initialized storage with collection: {COLLECTION_NAME}")
            
        except Exception as e:
            print(f"Failed to initialize storage: {e}")
            raise Exception(f"Storage initialization failed: {e}")

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
            update_result = self.collection.update(
                {"_key": document_id, "data.user": self.user},
                {"data": updated_data.model_dump(by_alias=True, mode="json"), "first_query": first_query}
            )
            if update_result:
                return "Updated document : {}".format(document_id)
            else:
                raise Exception("Not able to Update the Document")

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
            cursor = self.collection.find({"data.user": self.user}, {"data": 0})
            for document in cursor:
                document["id"] = document["_key"]
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
            print(response)
            breakpoint()
            if response and response["user"] == self.user:
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
            if doc and doc["user"] == self.user:
                self.collection.delete(document_id)
                return "Deleted document : {}".format(document_id)
            else:
                raise Exception("Not able to delete the Document")

        except Exception as e:
            print(e)
            raise Exception(e)

if __name__ == "__main__":
    import bson.errors as BsonError
    from bson.objectid import ObjectId
    from config import COLLECTION_NAME
    from arango_conn import ArangoClient
    from pydantic import BaseModel
    import asyncio
    print("Starting tests...")
    store = DocumentStore("test_user")
    store.initialize_storage()
    breakpoint()
    test_doc = TestDocument(
        messages="test message",
        user="test_user"
    )
        
    # Test save document
    def test_save():
        doc_id = store.save_document(test_doc)
        print(f"Saved document ID: {doc_id}")
        return doc_id
        
    # Test get document
    def test_get(doc_id):
        doc = store.get_user_documents_by_id(doc_id)
        print(f"Retrieved document: {doc}")
        return doc
        
    # Test delete document  
    def test_delete(doc_id):
        result = store.delete_document(doc_id)
        print(f"Delete result: {result}")
        
    def run_tests():
        # Run save test
        doc_id = test_save()
        breakpoint()
        
        # Run get test
        print(test_get(doc_id))
        breakpoint()
        # Run delete test
        print(test_delete(doc_id))
        breakpoint()
    run_tests()
 
