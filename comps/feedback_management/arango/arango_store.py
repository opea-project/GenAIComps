# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from arango_conn import ArangoClient
from config import COLLECTION_NAME
from pydantic import BaseModel


class FeedbackStore:

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

    def save_feedback(self, feedback_data: BaseModel) -> str:
        """Stores a new feedback data into the storage.

        Args:
            feedback_data (object): The document to be stored.

        Returns:
            str: The ID of the inserted feedback data.

        Raises:
            Exception: If an error occurs while storing the feedback_data.
        """
        try:
            model_dump = feedback_data.model_dump(by_alias=True, mode="json", exclude={"feedback_id"})

            inserted_feedback_data = self.collection.insert(model_dump)

            feedback_id = str(inserted_feedback_data["_key"])

            return feedback_id

        except Exception as e:
            print(e)
            raise Exception(e)

    def update_feedback(self, feedback_data: BaseModel) -> bool:
        """Update a feedback data in the collection with given id.

        Args:
            feedback_id (str): The ID of the data to be updated.
            updated_data (object):  The data to be updated in the entry.

        Returns:
            bool: True if the data updated successfully, False otherwise.

        Raises:
            KeyError: If the document with ID is not found.
            Exception: If the user does not match with the document user.
            Exception: If an error occurs while updating the feedback data.
        """
        _key = feedback_data.feedback_id
        document = self.collection.get(_key)

        if document is None:
            raise KeyError(f"Document with ID: {_key} not found.")

        if document["chat_data"]["user"] != self.user:
            raise Exception(f"User mismatch. Document with ID: {_key} does not belong to user: {self.user}")

        try:
            model_dump = feedback_data.feedback_data.model_dump(by_alias=True, mode="json")

            self.collection.update(
                {"_key": _key, "feedback_data": model_dump},
                merge=True,
                keep_none=False,
            )

            print(f"Updated document: {_key} !")

            return True

        except Exception as e:
            print("Not able to update the data.")
            print(e)
            raise Exception(e)

    def get_all_feedback_of_user(self) -> list[dict]:
        """Retrieves all feedback data of a user from the collection.

        Returns:
            list[dict] | None: List of dict of feedback data of the user, None otherwise.

        Raises:
            Exception: If there is an error while retrieving data.
        """
        try:
            feedback_data_list: list = []

            # TODO: Clarify if we actually want to omit the `feedback_data` field.
            # Implemented using MongoDB Feedback Management as a reference.
            cursor = """
                FOR doc IN @@collection
                    FILTER doc.chat_data.user == @user
                    RETURN UNSET(doc, "feedback_data")
            """

            cursor = self.db_client.aql.execute(
                cursor, bind_vars={"@collection": self.collection.name, "user": self.user}
            )

            for document in cursor:
                document["feedback_id"] = str(document["_key"])
                del document["_id"]
                del document["_key"]
                del document["_rev"]

                feedback_data_list.append(document)

            return feedback_data_list

        except Exception as e:
            print(e)
            raise Exception(e)

    def get_feedback_by_id(self, feedback_id: str) -> dict | None:
        """Retrieves a user feedback data from the collection based on the given feedback ID.

        Args:
            feedback_id (str): The ID of the feedback data to retrieve.

        Returns:
            dict | None: The user's feedback data if found, None otherwise.

        Raises:
            KeyError: If document with ID is not found.
            Exception: If the user does not match with the document user.
        """
        response = self.collection.get(feedback_id)

        if response is None:
            raise KeyError(f"Feedback with ID: {feedback_id} not found.")

        if response["chat_data"]["user"] != self.user:
            raise Exception(f"User mismatch. Feedback with ID: {feedback_id} does not belong to user: {self.user}")

        del response["_id"]
        del response["_key"]
        del response["_rev"]

        return response

    def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a feedback data from collection by given feedback_id.

        Args:
            feedback_id(str): The ID of the feedback data to be deleted.

        Returns:
            bool: True if feedback is successfully deleted, False otherwise.

        Raises:
            KeyError: If the provided feedback_id is invalid:
            Exception: If the user does not match with the document user.
            Exception: If any errors occurs during delete process.
        """
        response = self.collection.get(feedback_id)

        if response is None:
            raise KeyError(f"Feedback with ID: {feedback_id} not found.")

        if response["chat_data"]["user"] != self.user:
            raise Exception(f"User mismatch. Feedback with ID: {feedback_id} does not belong to user: {self.user}")

        try:
            response = self.collection.delete(feedback_id)
            print(f"Deleted document: {feedback_id} !")

            return True
        except Exception as e:
            print(e)
            raise Exception("Not able to delete the data.")
