from ..common.storage import OpeaStore
from ..mega.logger import CustomLogger

logger = CustomLogger("ArangoDBStore")


class ArangoDBStore(OpeaStore):
    """
    A concrete implementation of OpeaStore for ArangoDB.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        """
        Initializes the ArangoDBStore with the given configuration.

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

        try:
            from arango import ArangoClient
            from arango.collection import StandardCollection
            from arango.database import StandardDatabase
        except ImportError:
            logger.error(
                "ArangoDB client library is not installed. Please install it using 'pip install python-arango'."
            )
            raise

        super().__init__(name, description, config)

        self.client: ArangoClient = config.get("client", None)
        self.db: StandardDatabase = config.get("db", None)
        self.collection: StandardCollection = config.get("collection", None)

        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """
        Initializes the connection to the ArangoDB database and collection.
        """

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
                self.db = self.client.db(database_name, username=username, password=password)

            try:
                self.db.version()
            except Exception as e:
                logger.error(f"Failed to connect to ArangoDB database: {e}")
                raise

            if not self.collection:
                if not self.db.has_collection(collection_name):
                    self.collection = self.db.create_collection(collection_name)
                else:
                    self.collection = self.db.collection(collection_name)

            logger.info(f"Connected to ArangoDB database '{database_name}' and collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize ArangoDB connection: {e}")
            raise

    def save_document(self, doc: dict, **kwargs) -> bool | dict:
        """
        Save a single document to the store.
        Document can optionally contain a unique identifier.

        Args:
            doc (dict): The document data to save.
            **kwargs: Additional arguments for saving the document.
        """
        try:
            return self.collection.insert(doc, **kwargs)
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            raise

    def save_documents(self, docs: list[dict], **kwargs) -> bool | list:
        """
        Save multiple documents to the store.
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
        """
        try:
            return self.collection.insert_many(docs, **kwargs)
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
            raise

    def update_document(self, doc: dict, **kwargs) -> bool | dict:
        """
        Update a single document in the store.
        Document must contain its unique identifier.

        Args:
            doc (dict): The document data to update.
        """
        try:
            return self.collection.update(doc, **kwargs)
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            raise

    def update_documents(self, docs: list[dict], **kwargs) -> bool | list:
        """
        Update multiple documents in the store.
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
        """
        try:
            return self.collection.update_many(docs, **kwargs)
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            raise

    def get_document_by_id(self, id: str, **kwargs) -> dict | None:
        """
        Retrieve a single document by its unique identifier.

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
        """
        Retrieve multiple documents by their unique identifiers.

        Args:
            ids (list[str]): A list of unique identifiers for the documents.

        Returns:
            list[dict]: A list of retrieved document data.
        """
        try:
            return self.collection.get_many(ids, **kwargs)
        except Exception as e:
            logger.error(f"Failed to retrieve documents by IDs {ids}: {e}")
            raise

    def delete_document(self, id: str, **kwargs) -> bool | dict:
        """
        Delete a single document from the store.

        Args:
            id (str): The unique identifier for the document.
        """
        try:
            return self.collection.delete(id)
        except Exception as e:
            logger.error(f"Failed to delete document by ID {id}: {e}")
            raise

    def delete_documents(self, ids: list[str], **kwargs) -> bool | list:
        """
        Delete multiple documents from the store.

        NOTE: If updating a document fails, the exception is not raised
        but returned as an object in the result list. It is up to you to
        inspect the list to determine which documents were
        updated successfully (returns document metadata)
        and which were not (returns exception object).
        Alternatively, you can rely on setting
        raise_on_document_error to True (defaults to False).

        Args:
            ids (list[str]): A list of unique identifiers for the documents.
        """
        try:
            return self.collection.delete_many(ids, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete documents by IDs {ids}: {e}")
            raise
