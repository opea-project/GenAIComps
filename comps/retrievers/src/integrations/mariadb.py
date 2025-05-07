# Copyright (C) 2025 MariaDB Foundation
# SPDX-License-Identifier: Apache-2.0


import os
from urllib.parse import urlparse

import mariadb
from fastapi import HTTPException
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mariadb.vectorstores import MariaDBStore, MariaDBStoreSettings

from comps import CustomLogger, EmbedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType

from .config import (
    EMBED_MODEL,
    HUGGINGFACEHUB_API_TOKEN,
    MARIADB_COLLECTION_NAME,
    MARIADB_CONNECTION_URL,
    TEI_EMBEDDING_ENDPOINT,
)


class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


logger = CustomLogger("mariadbvector_retrievers")
logflag = os.getenv("LOGFLAG", False)
if not logflag:
    logger = NullLogger()


@OpeaComponentRegistry.register("OPEA_RETRIEVER_MARIADBVECTOR")
class OpeaMARIADBVectorRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for mariadb vector retriever services.

    Attributes:
        client (MariaDBStore): An instance of the MariaDBStore client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        args = urlparse(MARIADB_CONNECTION_URL)

        self.conn_args = {
            "host": args.hostname,
            "port": args.port,
            "user": args.username,
            "password": args.password,
            "database": args.path[1:],
        }

        self.embedder = self._initialize_embedder()

        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaMARIADBVectorRetriever health check failed.")

        self.store = self._initialize_client()

    def _initialize_embedder(self):
        if TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            logger.info(f"[ init embedder ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
            if not HUGGINGFACEHUB_API_TOKEN:
                raise HTTPException(
                    status_code=400,
                    detail="You MUST offer the `HUGGINGFACEHUB_API_TOKEN` when using `TEI_EMBEDDING_ENDPOINT`.",
                )
            import requests

            response = requests.get(TEI_EMBEDDING_ENDPOINT + "/info")
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available."
                )
            model_id = response.json()["model_id"]
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=HUGGINGFACEHUB_API_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        else:
            # create embeddings using local embedding model
            logger.info(f"[ init embedder ] LOCAL_EMBEDDING_MODEL:{EMBED_MODEL}")
            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return embeddings

    def _initialize_client(self) -> MariaDBStore:
        store = MariaDBStore(
            embeddings=self.embedder,
            collection_name=MARIADB_COLLECTION_NAME,
            datasource=MARIADB_CONNECTION_URL,
            config=MariaDBStoreSettings(lazy_init=True),
        )
        return store

    def check_health(self) -> bool:
        """Checks mariadb server health."""
        try:
            connection = mariadb.connect(**self.conn_args)
            return True
        except mariadb.Error as e:
            logger.error(f"Error connect to MariaDB Server: {e}")
            return False

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return False
        finally:
            try:
                connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

    async def invoke(self, input: EmbedDoc) -> list:
        """Search the MariaDB Vector index for the most similar documents to the input query.

        Args:
            input (EmbedDoc): The input query to search for.
        Output:
            list: The retrieved documents.
        """
        logger.info(f"[ similarity search ] input: {input}")

        result = []
        try:
            result = await self.store.asimilarity_search_by_vector(embedding=input.embedding)
            logger.info(f"[ similarity search ] search result: {result}")
            return result
        except mariadb.Error as e:
            logger.error(f"A database error occurred during similarity search: {e}")
            raise HTTPException(status_code=500, detail="A database error occurred during similarity search")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred")
