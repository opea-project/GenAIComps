# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from arango.exceptions import IndexGetError
from arango_conn import ArangoClient
from config import COLLECTION_NAME
from prompt import PromptCreate
from pydantic import BaseModel

from comps import CustomLogger

logger = CustomLogger("arango_store")
logflag = os.getenv("LOGFLAG", False)


class PromptStore:

    def __init__(
        self,
        user: str,
    ):
        self.user = user
        self.inverted_index_exists = False

    def initialize_storage(self) -> None:
        self.db_client = ArangoClient.get_db_client()

        if not self.db_client.has_collection(COLLECTION_NAME):
            self.db_client.create_collection(COLLECTION_NAME)

        self.collection = self.db_client.collection(COLLECTION_NAME)

    def save_prompt(self, prompt: PromptCreate):
        """Stores a new prompt into the storage.

        Args:
            prompt: The document to be stored. It should be a Pydantic model.

        Returns:
            str: The ID of the inserted prompt.

        Raises:
            Exception: If an error occurs while storing the prompt.
        """
        try:
            model_dump = prompt.model_dump(by_alias=True, mode="json", exclude={"id"})

            inserted_prompt_data = self.collection.insert(model_dump)

            prompt_id = str(inserted_prompt_data["_key"])

            return prompt_id

        except Exception as e:
            print(e)
            raise Exception(e)

    def get_all_prompt_of_user(self) -> list[dict]:
        """Retrieves all prompts of a user from the collection.

        Returns:
            list[dict] | None: List of dict of prompts of the user, None otherwise.

        Raises:
            Exception: If there is an error while retrieving data.
        """
        try:
            prompt_data_list: list = []

            # TODO: Clarify if we actually want to omit the `data` field.
            # Implemented using MongoDB Prompt Registry as a reference.
            cursor = """
                FOR doc IN @@collection
                    FILTER doc.chat_data.user == @user
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

                prompt_data_list.append(document)

            return prompt_data_list

        except Exception as e:
            print(e)
            raise Exception(e)

    def get_user_prompt_by_id(self, prompt_id: str) -> dict | None:
        """Retrieves a user prompt from the collection based on the given prompt ID.

        Args:
            prompt_id (str): The ID of the prompt to retrieve.

        Returns:
            dict | None: The user prompt if found, None otherwise.

        Raises:
            KeyError: If document with ID is not found.
            Exception: If the user does not match with the document user.
        """
        response = self.collection.get(prompt_id)

        if response is None:
            raise KeyError(f"Prompt with ID: {prompt_id} not found.")

        if response["user"] != self.user:
            raise Exception(f"User mismatch. Prompt with ID: {prompt_id} does not belong to user: {self.user}")

        del response["_id"]
        del response["_key"]
        del response["_rev"]

        return response

    def prompt_search(self, keyword: str) -> list | None:
        """Retrieves prompt from the collection based on keyword provided.

        Args:
            keyword (str): The keyword of prompt to search for.

        Returns:
            list | None: The list of relevant prompt if found, None otherwise.

        Raises:
            Exception: If there is an error while searching data.
        """
        try:
            index_name = "prompt_text_index"

            if not self.inverted_index_exists:
                try:
                    self.collection.get_index(index_name)

                except IndexGetError:
                    self.collection.add_inverted_index(
                        fields=["prompt_text"],
                        name=index_name,
                        # TODO: add more kwargs if needed
                    )

                self.inverted_index_exists = True

            query = """
                FOR doc IN @@collection
                OPTIONS { indexHint: @index_name, forceIndexHint: true }
                    FILTER PHRASE(doc.prompt_text, @keyword, "text_en")
                    RETURN doc
            """

            cursor = self.db_client.aql.execute(
                query,
                bind_vars={
                    "@collection": self.collection.name,
                    "index_name": index_name,
                    "keyword": keyword,
                },
            )

            serialized_data = []
            for doc in cursor:
                doc["id"] = str(doc["_key"])
                del doc["_id"]
                del doc["_key"]
                del doc["_rev"]

                serialized_data.append(doc)

            return serialized_data

        except Exception as e:
            print(e)
            raise Exception(e)

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt from collection by given prompt_id.

        Args:
            prompt_id(str): The ID of the prompt to be deleted.

        Returns:
            bool: True if prompt is successfully deleted, False otherwise.

        Raises:
            KeyError: If the provided feedback_id is invalid:
            Exception: If the user does not match with the document user.
            Exception: If any errors occurs during delete process.
        """
        response = self.collection.get(prompt_id)

        if response is None:
            raise KeyError(f"Feedback with ID: {prompt_id} not found.")

        if response["user"] != self.user:
            raise Exception(f"User mismatch. Feedback with ID: {prompt_id} does not belong to user: {self.user}")

        try:
            response = self.collection.delete(prompt_id)
            print(f"Deleted document: {prompt_id} !")

            return True
        except Exception as e:
            print(e)
            raise Exception("Not able to delete the data.")
