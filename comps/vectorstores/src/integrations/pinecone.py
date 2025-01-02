# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import time
from typing import List, Optional

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from comps import CustomLogger, OpeaComponent, ServiceType

from .config import PINECONE_API_KEY, PINECONE_INDEX_NAME

logger = CustomLogger("pinecone_vectorstores")
logflag = os.getenv("LOGFLAG", False)


class OpeaPineconeVectorstores(OpeaComponent):

    def __init__(
        self,
        embedder,
        name: str,
        description: str,
        config: dict = None,
        pinecone_api_key: str = PINECONE_API_KEY,
        pinecone_index: str = PINECONE_INDEX_NAME,
    ):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.embedder = embedder
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index = pinecone_index
        self.pc, self.index, self.vector_db = self._initialize_vector_db()

    def _initialize_vector_db(self) -> Pinecone:
        """ "Initialize the Pinecone vector db client."""
        pc = Pinecone(api_key=self.pinecone_api_key)
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if self.pinecone_index in existing_indexes:
            pc.configure_index(self.pinecone_index, deletion_protection="disabled")
            pc.delete_index(self.pinecone_index)
            time.sleep(1)

        pc.create_index(
            self.pinecone_index,
            dimension=768,
            deletion_protection="disabled",
            spec=spec,
        )
        while not pc.describe_index(self.pinecone_index).status["ready"]:
            time.sleep(1)

        index = pc.Index(self.pinecone_index)
        vector_db = PineconeVectorStore(index=index, embedding=self.embedder)
        return pc, index, vector_db

    def check_health(self) -> bool:
        """Checks the health of the dataprep service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ check health ] start to check health of pinecone")
        try:
            # Check the status of the Pinecone service
            health_status = self.index.describe_index_stats()
            logger.info(f"[ check health ] health status: {health_status}")
            logger.info("[ check health ] Successfully connected to Pinecone!")
            return True
        except Exception as e:
            logger.info(f"[ check health ] Failed to connect to Pinecone: {e}")
            return False

    def invoke(self, *args, **kwargs):
        pass

    def is_empty(self) -> bool:
        """Check whether the database is empty.

        Returns:
            bool: True if the database is empty, False otherwise
        """
        total_count = self.index.describe_index_stats()["total_vector_count"]
        if logflag:
            logger.info(f"[ is empty ] total count: {total_count}")
        return total_count == 0

    async def ingest_chunks(self, file_name: str, chunks: List, batch_size: int = 32) -> bool:
        """Ingest string chunks into pinecone database.

        Args:
            file_name (str): The name of the file.
            chunks (List): The list of string chunks.
            batch_size (int): The batch size for ingesting chunks into db.
        Returns:
            bool: True if the chunks are ingested successfully, False otherwise.
        """
        if logflag:
            logger.info(f"[ ingest chunks ] file name:{file_name}")

        # Checking Index existence
        if not self.check_index_existance():
            # Creating the index
            self.create_index()
            if logflag:
                logger.info(f"[ ingest chunks ] Successfully created the index {self.pinecone_index}")

        # Ingesting chunks
        num_chunks = len(chunks)

        for i in range(0, num_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = batch_chunks

            try:
                _ = PineconeVectorStore.from_texts(
                    texts=batch_texts,
                    embedding=self.embedder,
                    index_name=self.pinecone_index,
                )
                if logflag:
                    logger.info(
                        f"[ ingest chunks ] Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}"
                    )
            except Exception as e:
                if logflag:
                    logger.error(f"[ ingest chunks ] fail to ingest chunks into Pinecone. error: {e}")
                return False

        return True

    async def delete_all_files(self) -> bool:
        """Delete all files in pinecone database.

        Returns:
            bool: True if all files are deleted successfully, False otherwise.
        """
        if logflag:
            logger.info("[ delete all files ] delete all files")

        try:
            self.pc.delete_index(self.pinecone_index)
            if logflag:
                logger.info(f"[ drop index ] index {self.pinecone_index} deleted")
        except Exception as e:
            if logflag:
                logger.info(f"[ drop index ] index {self.pinecone_index} delete failed: {e}")
            return False
        return True

    async def similarity_search(
        self,
        input: str,
        embedding: list,
        search_type: str = "similarity",
        k: int = 4,
        fetch_k: int = 20,
        distance_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
        lambda_mult: float = 0.2,
    ):
        if logflag:
            logger.info(f"[ similarity search ] search type: {search_type}, input: {input}")

        if search_type == "similarity":
            docs_and_similarities = await self.vector_db.similarity_search_by_vector_with_score(
                embedding=embedding, k=k
            )
            search_res = [doc for doc, _ in docs_and_similarities]
        elif search_type == "similarity_distance_threshold":
            if distance_threshold is None:
                raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
            docs_and_similarities = self.vector_db.similarity_search_by_vector_with_score(embedding=embedding, k=k)
            search_res = [doc for doc, similarity in docs_and_similarities if similarity > distance_threshold]
        elif search_type == "similarity_score_threshold":
            docs_and_similarities = self.vector_db.similarity_search_by_vector_with_score(query=input, k=k)
            search_res = [doc for doc, similarity in docs_and_similarities if similarity > score_threshold]
        elif search_type == "mmr":
            search_res = await self.vector_db.max_marginal_relevance_search(
                query=input, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
        else:
            search_res = self.vector_db.max_marginal_relevance_search(query=input, k=k, fetch_k=fetch_k)

        if logflag:
            logger.info(f"[ similarity search ] search result: {search_res}")
        return search_res

    ###############################
    # Pinecone specific functions #
    ###############################
    def check_index_existance(self) -> bool:
        if logflag:
            logger.info(f"[ check index existence ] checking {self.pinecone_index} index existence")
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if self.pinecone_index not in existing_indexes:
            if logflag:
                logger.info("[ check index existence ] index does not exist")
            return False
        else:
            return True

    def create_index(self) -> bool:
        if logflag:
            logger.info(f"[ create index ] creating index {self.pinecone_index}")
        try:
            self.pc.create_index(
                name=self.pinecone_index,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            if logflag:
                logger.info(f"[ create index ] index {self.pinecone_index} successfully created")
        except Exception as e:
            if logflag:
                logger.info(f"[ create index ] fail to create index {self.pinecone_index}: {e}")
            return False
        return True
