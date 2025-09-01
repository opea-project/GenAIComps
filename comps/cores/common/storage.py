# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

from ..mega.logger import CustomLogger

logger = CustomLogger("OpeaStore")


class OpeaStore(ABC):
    """The OpeaStore class serves as the base class for all Storage APIs.
    It provides a unified interface and foundational attributes that every derived Storage API inherits and extends.

    Attributes:
        name (str): The name of the component (e.g 'arangodb', 'redis', 'mongodb', etc.)
        description (str): A brief description of the component's functionality.
        config (dict): A dictionary containing configuration parameters for the component.
    """

    def __init__(self, name: str, description: str = "", config: dict = {}):
        """Initializes an OpeaComponent instance with the provided attributes.

        Args:
            name (str): The name of the component.
            description (str): A brief description of the component.
            config (dict, optional): Configuration parameters for the component. Defaults to an empty dictionary.
        """
        self.name = name
        self.description = description
        self.config = config if config is not None else {}

    def get_meta(self) -> dict:
        """Retrieves metadata about the component, including its name, type, description, and configuration.

        Returns:
            dict: A dictionary containing the component's metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config,
        }

    def update_config(self, key: str, value):
        """Updates a configuration parameter for the component.

        Args:
            key (str): The configuration parameter's key.
            value: The new value for the configuration parameter.
        """
        self.config[key] = value

    def health_check(self) -> bool:
        """Performs a health check on the component to ensure it is connected to
        the database correctly.

        Returns:
            bool: True if the component is healthy, False otherwise.
        """
        raise NotImplementedError("health_check method must be implemented by subclasses.")

    def __repr__(self):
        """Provides a string representation of the component for debugging and logging purposes.

        Returns:
            str: A string representation of the OpeaComponent instance.
        """
        return f"OpeaStore(name={self.name}, type={self.type}, description={self.description})"

    def save_document(self, doc: dict) -> None:
        """Save a single document to the store.
        Document can optionally contain a unique identifier.

        Args:
            doc (dict): The document data to save.
        """
        raise NotImplementedError("save_document method must be implemented by subclasses.")

    async def asave_document(self, doc: dict) -> None:
        """Asynchronously save a single document to the store.
        Document can optionally contain a unique identifier.

        Args:
            doc (dict): The document data to save.
        """
        raise NotImplementedError("asave_document method must be implemented by subclasses.")

    def save_documents(self, docs: list[dict]) -> None:
        """Save multiple documents to the store.
        Documents can optionally contain unique identifiers.

        Args:
            docs (list[dict]): A list of document data to save.
        """
        raise NotImplementedError("save_documents method must be implemented by subclasses.")

    async def asave_documents(self, docs: list[dict]) -> None:
        """Asynchronously save multiple documents to the store.
        Documents can optionally contain unique identifiers.

        Args:
            docs (list[dict]): A list of document data to save.
        """
        raise NotImplementedError("asave_documents method must be implemented by subclasses.")

    def update_document(self, doc: dict) -> None:
        """Update a single document in the store.
        Document must contain its unique identifier.

        Args:
            doc (dict): The document data to update.
        """
        raise NotImplementedError("update_document method must be implemented by subclasses.")

    async def aupdate_document(self, doc: dict) -> None:
        """Asynchronously update a single document in the store.
        Document must contain its unique identifier.

        Args:
            doc (dict): The document data to update.
        """
        raise NotImplementedError("aupdate_document method must be implemented by subclasses.")

    def update_documents(self, docs: list[dict]) -> None:
        """Update multiple documents in the store.
        Each document must contain its unique identifier.

        Args:
            docs (list[dict]): The list of documents to update.
        """
        raise NotImplementedError("update_documents method must be implemented by subclasses.")

    async def aupdate_documents(self, docs: list[dict]) -> None:
        """Asynchronously update multiple documents in the store.
        Each document must contain its unique identifier.

        Args:
            docs (list[dict]): The list of documents to update.
        """
        raise NotImplementedError("aupdate_documents method must be implemented by subclasses.")

    def get_document_by_id(self, id: str) -> dict:
        """Retrieve a single document by its unique identifier.

        Args:
            id (str): The unique identifier for the document.

        Returns:
            dict: The retrieved document data.
        """
        raise NotImplementedError("get_document_by_id method must be implemented by subclasses.")

    async def aget_document_by_id(self, id: str) -> dict:
        """Asynchronously retrieve a single document by its unique identifier.

        Args:
            id (str): The unique identifier for the document.

        Returns:
            dict: The retrieved document data.
        """
        raise NotImplementedError("aget_document_by_id method must be implemented by subclasses.")

    def get_documents_by_ids(self, ids: list[str]) -> list[dict]:
        """Retrieve multiple documents by their unique identifiers.

        Args:
            ids (list[str]): A list of unique identifiers for the documents.

        Returns:
            list[dict]: A list of retrieved document data.
        """
        raise NotImplementedError("get_documents_by_ids method must be implemented by subclasses.")

    async def aget_documents_by_ids(self, ids: list[str]) -> list[dict]:
        """Asynchronously retrieve multiple documents by their unique identifiers.

        Args:
            ids (list[str]): A list of unique identifiers for the documents.

        Returns:
            list[dict]: A list of retrieved document data.
        """
        raise NotImplementedError("aget_documents_by_ids method must be implemented by subclasses.")

    def delete_document(self, id: str) -> None:
        """Delete a single document from the store.

        Args:
            id (str): The unique identifier for the document.
        """
        raise NotImplementedError("delete_document method must be implemented by subclasses.")

    async def adelete_document(self, id: str) -> None:
        """Asynchronously delete a single document from the store.

        Args:
            id (str): The unique identifier for the document.
        """
        raise NotImplementedError("adelete_document method must be implemented by subclasses.")

    def delete_documents(self, ids: list[str]) -> None:
        """Delete multiple documents from the store.

        Args:
            ids (list[str]): A list of unique identifiers for the documents.
        """
        raise NotImplementedError("delete_documents method must be implemented by subclasses.")

    async def adelete_documents(self, ids: list[str]) -> None:
        """Asynchronously delete multiple documents from the store.

        Args:
            ids (list[str]): A list of unique identifiers for the documents.
        """
        raise NotImplementedError("adelete_documents method must be implemented by subclasses.")

    def search(self, key: str, value: Any, search_type: str = "exact", **kwargs) -> list[dict]:
        """Search for documents in the store based on a specific key-value pair.

        Args:
            key (str): The key to search for.
            value (str): The value to search for.
            search_type (str): The type of search to perform.
                Can be ignored for some implementations.
            **kwargs: Additional arguments for the search query.

        Returns:
            list[dict]: A list of documents matching the search criteria.
        """
        raise NotImplementedError("search_by_keyword method must be implemented by subclasses.")

    async def asearch(self, key: str, value: Any, search_type: str = "exact", **kwargs) -> list[dict]:
        """Asynchronously search for documents in the store based on a specific key-value pair.

        Args:
            key (str): The key to search for.
            value (str): The value to search for.
            search_type (str): The type of search to perform.
                Can be ignored for some implementations.
            **kwargs: Additional arguments for the search query.

        Returns:
            list[dict]: A list of documents matching the search criteria.
        """
        raise NotImplementedError("asearch_by_keyword method must be implemented by subclasses.")

    async def asearch_by_keyword(self, keyword: str, max_results: int = 5, **kwargs) -> list[dict]:
        """Asynchronously search for documents in the store based on a specific keyword.

        Args:
            keyword (str): The keyword to search for.
            **kwargs: Additional arguments for the search query.

        Returns:
            list[dict]: A list of documents matching the search criteria.
        """
        raise NotImplementedError("asearch_by_keyword method must be implemented by subclasses.")
