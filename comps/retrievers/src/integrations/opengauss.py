# Copyright (C) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


import os
from urllib.parse import urlparse

import psycopg2
from fastapi import HTTPException
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_opengauss import OpenGauss, OpenGaussSettings

from comps import CustomLogger, EmbedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType

from .config import EMBED_MODEL, GS_CONNECTION_STRING, GS_INDEX_NAME, HF_TOKEN, TEI_EMBEDDING_ENDPOINT

logger = CustomLogger("opengauss_retrievers")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_RETRIEVER_OPENGAUSS")
class OpeaOpenGaussRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for openGauss retriever services.

    Attributes:
        client (openGauss): An instance of the openGauss client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self.embedder = self._initialize_embedder()
        self.gs_connection_string = GS_CONNECTION_STRING
        self.gs_index_name = GS_INDEX_NAME
        self.vector_db = self._initialize_client()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaOpenGaussRetriever health check failed.")

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

            response = requests.get(TEI_EMBEDDING_ENDPOINT + "/info")
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available."
                )
            model_id = response.json()["model_id"]
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=HF_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        else:
            # create embeddings using local embedding model
            if logflag:
                logger.info(f"[ init embedder ] EMBEDDING_MODEL:{EMBED_MODEL}")
            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return embeddings

    def _initialize_client(self) -> OpenGauss:
        """Initializes the openGauss client."""
        result = urlparse(GS_CONNECTION_STRING)
        username = result.username
        password = result.password
        database = result.path[1:]
        hostname = result.hostname
        port = result.port

        self.opengauss_config = OpenGaussSettings(
            host=hostname, port=port, user=username, password=password, database=database, embedding_dimension=768
        )
        vector_db = OpenGauss(embedding=self.embedder, config=self.opengauss_config)
        return vector_db

    def check_health(self) -> bool:
        """Checks the health of the retriever service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        result = urlparse(GS_CONNECTION_STRING)
        username = result.username
        password = result.password
        database = result.path[1:]
        hostname = result.hostname
        port = result.port

        if logflag:
            logger.info("[ check health ] start to check health of openGauss")
        try:
            # Check the status of the openGauss service
            psycopg2.connect(database=database, user=username, password=password, host=hostname, port=port)
            logger.info("[ check health ] Successfully connected to openGauss!")
            return True
        except Exception as e:
            logger.info(f"[ check health ] Failed to connect to openGauss: {e}")
            return False

    async def invoke(self, input: EmbedDoc) -> list:
        """Search the openGauss index for the most similar documents to the input query.

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
