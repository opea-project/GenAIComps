# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pyodbc
from fastapi import HTTPException
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_sqlserver.vectorstores import SQLServer_VectorStore

from comps import CustomLogger, EmbedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType

from .config import EMBED_MODEL, HF_TOKEN, MSSQL_CONNECTION_STRING, TABLE_NAME, TEI_EMBEDDING_ENDPOINT

logger = CustomLogger("sqlserver_retrievers")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_RETRIEVER_SQLSERVER")
class OpeaSqlServerRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for SQL Server retriever services.

    Attributes:
        client (SQLServer): An instance of the SQLServer client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self.embedder = self._initialize_embedder()
        self.MSSQL_CONNECTION_STRING = MSSQL_CONNECTION_STRING
        self.sqlserver_table_name = TABLE_NAME
        self.vector_db = self._initialize_client()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaSqlServerRetriever health check failed.")

    def _initialize_embedder(self):
        if TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            if logflag:
                logger.info(f"[ init embedder ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
            if not HF_TOKEN:
                raise HTTPException(
                    status_code=400,
                    detail="You MUST offer the `HF_TOKEN` when using `TEI_EMBEDDING_ENDPOINT`.",
                )
            import requests

            logger.info(f"Attempting to contact TEI embedding endpoint: {TEI_EMBEDDING_ENDPOINT}/info")
            try:
                response = requests.get(TEI_EMBEDDING_ENDPOINT + "/info")
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400, detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available."
                    )
                model_id = response.json()["model_id"]
                logger.info(f"Using TEI embedding model: {model_id} from endpoint: {TEI_EMBEDDING_ENDPOINT}")

            except requests.RequestException as e:
                logger.error(f"Failed to contact TEI embedding endpoint: {TEI_EMBEDDING_ENDPOINT}. Error: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available or returned an error.",
                )

            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=HF_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        else:
            # create embeddings using local embedding model
            if logflag:
                logger.info(f"[ init embedder ] LOCAL_EMBEDDING_MODEL:{EMBED_MODEL}")
            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        # Determine embedding length.
        try:
            embedding = embeddings.embed_documents(["Test input text to get embedding size"])
            if not embedding or not isinstance(embedding[0], list):
                raise ValueError("Embedding generation returned an unexpected format.")
            self.embedding_length = len(embedding[0])
            logger.info(f"Embedding Length of the model: {self.embedding_length}")
        except Exception as e:
            logger.error(f"Failed to generate embedding for model '{EMBED_MODEL}': {e}")
            raise RuntimeError(
                "Embedding initialization failed. Please check the model configuration and embedding service."
            )

        return embeddings

    def _initialize_client(self) -> SQLServer_VectorStore:
        """Initializes the SQL server client."""
        vector_db = SQLServer_VectorStore(
            embedding_function=self.embedder,
            table_name=self.sqlserver_table_name,
            connection_string=self.MSSQL_CONNECTION_STRING,
            embedding_length=self.embedding_length,
        )
        return vector_db

    def check_health(self) -> bool:
        """Checks the health of the retriever service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ check health ] start to check health of SQL Server")
        try:
            conn = pyodbc.connect(MSSQL_CONNECTION_STRING)
            conn.close()
            logger.info("[ check health ] Successfully connected to SQL Server!")
            return True

        except pyodbc.Error as e:
            logger.error("Error connecting to MS SQL")

        return False

    async def invoke(self, input: EmbedDoc) -> list:
        """Search the SQLServer index for the most similar documents to the input query.

        Args:
            input (EmbedDoc): The input query to search for.
        Output:
            list: The retrieved documents.
        """
        if logflag:
            logger.info(f"[ similarity search ] input: {input}")

        search_res = await self.vector_db.asimilarity_search_by_vector(embedding=input.embedding)

        if logflag:
            logger.info(f"[ similarity search ] search result: {search_res}")
        return search_res
