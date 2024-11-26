from typing import Any

from arango_conn import ArangoClient
from config import COLLECTION_NAME
from pydantic import BaseModel


class DocumentStore:

    def __init__(
        self,
        user: str,
    ):
        self.user = user

    def initialize_storage(self) -> None:
        self.db_client = ArangoClient.get_db_client()

        if not self.db_client.has_collection(COLLECTION_NAME):
            self.db_client.create_collection(COLLECTION_NAME)

        self.collection = self.db_client.collection(COLLECTION_NAME)

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
            model_dump = document.model_dump(by_alias=True, mode="json", exclude={"id"})

            inserted_document = self.collection.insert(model_dump)

            document_id = str(inserted_document["_key"])

            return document_id

        except Exception as e:
            print(e)
            raise Exception(e)

    def update_document(self, document_id: str, updated_data: BaseModel, first_query: Any) -> str:
        """Updates a document in the collection with the given document_id.

        Args:
            document_id (str): The ID of the document to update.
            updated_data (object): The updated data to be set in the document.
            first_query (object): The first query to be set in the document.

        Returns:
            bool: True if the document was successfully updated, False otherwise.

        Raises:
            KeyError: If the document with ID is not found.
            Exception: If the user does not match with the document user.
            Exception: If an error occurs while updating the document data.
        """
        document = self.collection.get(document_id)

        if document is None:
            raise Exception(f"Unable to find Document {document_id}")

        if document["data"]["user"] != self.user:
            raise Exception(f"User {self.user} is not allowed to update Document {document_id}.")

        try:
            self.collection.update(
                {
                    "_key": document_id,
                    "data": updated_data.model_dump(by_alias=True, mode="json"),
                    "first_query": first_query,
                },
                merge=True,
                keep_none=True,
            )

            print(f"Updated document: {document_id} !")

            return True

        except Exception as e:
            print("Not able to update the data.")
            print(e)
            raise Exception(e)

    def get_all_documents_of_user(self) -> list[dict]:
        """Retrieves all documents of a specific user from the collection.

        Returns:
            A list of dictionaries representing the conversation documents.
        Raises:
            Exception: If there is an error while retrieving the documents.
        """
        try:
            document_list: list = []

            # TODO: Clarify if we actually want to omit the `data` field.
            # Implemented using MongoDB Feedback Management as a reference.
            cursor = """
                FOR doc IN @@collection
                    FILTER doc.data.user == @user
                    RETURN UNSET(doc, "data")
            """

            cursor = self.db_client.aql.execute(
                cursor, bind_vars={"@collection": self.collection.name, "user": self.user}
            )

            for document in cursor:
                document["id"] = str(document["_key"])
                del document["_id"]
                del document["_key"]
                del document["_rev"]

                document_list.append(document)

            return document_list

        except Exception as e:
            print(e)
            raise Exception(e)

    def get_user_documents_by_id(self, document_id: str) -> dict | None:
        """Retrieves a user document from the collection based on the given document ID.

        Args:
            document_id (str): The ID of the document to retrieve.

        Returns:
            dict | None: The user document if found, None otherwise.

        Raises:
            KeyError: If document with ID is not found.
            Exception: If the user does not match with the document user.
        """
        response = self.collection.get(document_id)

        if response is None:
            raise KeyError(f"Document with ID: {document_id} not found.")

        if response["data"]["user"] != self.user:
            raise Exception(f"User mismatch. Document with ID: {document_id} does not belong to user: {self.user}")

        del response["_id"]
        del response["_key"]
        del response["_rev"]

        return response

    def delete_document(self, document_id: str) -> str:
        """Deletes a document from the collection based on the provided document ID.

        Args:
            document_id (str): The ID of the document to be deleted.

        Returns:
            bool: True if the document is successfully deleted, False otherwise.

        Raises:
            KeyError: If the provided document_id is invalid:
            Exception: If the user does not match with the document user.
            Exception: If any errors occurs during delete process.
        """
        response = self.collection.get(document_id)

        if response is None:
            raise KeyError(f"Document with ID: {document_id} not found.")

        if response["data"]["user"] != self.user:
            raise Exception(f"User mismatch. Feedback with ID: {document_id} does not belong to user: {self.user}")

        try:
            response = self.collection.delete(document_id)
            print(f"Deleted document: {document_id} !")

            return True
        except Exception as e:
            print(e)
            raise Exception("Not able to delete the data.")
