# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import redis
from typing import List, Optional
from .config import INDEX_NAME, KEY_INDEX_NAME, REDIS_URL, SEARCH_BATCH_SIZE, INDEX_SCHEMA
from langchain_community.vectorstores import Redis
from redis.commands.search.field import TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from comps import OpeaComponent, CustomLogger, ServiceType
from comps.vectorstores.src.utils import encode_filename, format_search_results

logger = CustomLogger("redis_vectorstores")
logflag = os.getenv("LOGFLAG", False)


class OpeaRedisVectorstores(OpeaComponent):

    def __init__(
            self,
            embedder,
            name: str, 
            description: str, 
            config: dict = None, 
            redis_url: str=REDIS_URL,
            index_name: str=INDEX_NAME,
            key_index_name: str=KEY_INDEX_NAME,
            search_batch_size: int=SEARCH_BATCH_SIZE,
            is_multimodal: bool=False
            ):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.embedder = embedder
        self.redis_url = redis_url
        self.redis_pool = redis.ConnectionPool.from_url(self.redis_url)
        logger.info(f"redis url: {self.redis_url}\nredis pool: {self.redis_pool}")
        self.data_index_name = index_name
        self.key_index_name = key_index_name
        self.search_batch_size = search_batch_size
        self.client = self._initialize_client()
        self.data_index_client = self.client.ft(self.data_index_name)
        self.key_index_client = self.client.ft(self.key_index_name)
        self.is_multimodal = is_multimodal
        self.vector_db = self._initialize_vector_db()

    def _initialize_client(self) -> redis.Redis:
        """Initializes the redis client."""
        if logflag:
            logger.info(f"[ initialize client ] initializing redis client...")

        try: 
            logger.info(f"redis pool: {self.redis_pool}")
            client = redis.Redis(connection_pool=self.redis_pool)
            return client
        except Exception as e:
            logger.error(f"[ initialize client ] fail to initialize redis client: {e}")
            return None

    def _initialize_vector_db(self) -> Redis:
        """"Initialize the redis vector db client."""
        if self.is_multimodal:
            logger.info(f"[ initialize vectordb ] multimodal")
            vectordb = Redis(embedding=self.embedder, index_name=self.data_index_name, index_schema=INDEX_SCHEMA, redis_url=self.redis_url)
        else:
            vectordb = Redis(embedding=self.embedder, index_name=self.data_index_name, redis_url=self.redis_url)
        return vectordb
    
    def check_health(self) -> bool:
        """
        Checks the health of the dataprep service.
        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info(f"[ check health ] start to check health of redis")
        try:
            if self.client.ping():
                logger.info("[ check health ] Successfully connected to Redis!")
                return True
        except redis.ConnectionError as e:
            logger.info(f"[ check health ] Failed to connect to Redis: {e}")
            return False

    def invoke(self, *args, **kwargs):
        pass

    def is_empty(self):
        """
        Check whether the redis db is empty.
        Returns:
            True if redis is empty, False otherwise.
        """
        dbsize = self.client.dbsize()
        if logflag:
            logger.info(f"[ is empty ] redis db size: {dbsize}")
        return dbsize == 0

    async def ingest_chunks(
            self, 
            file_name: str, 
            chunks: List,
            batch_size: int=32
        ) -> bool:
        """
        Ingest string chunks into redis database.
        Args:
            file_name (str): The name of the file.
            chunks (List): The list of string chunks.
            batch_size (int): The batch size for ingesting chunks into db.
        Returns:
            bool: True if the chunks are ingested successfully, False otherwise.
        """
        if logflag:
            logger.info(f"[ ingest ] file name:{file_name}")

        # Batch size
        num_chunks = len(chunks)

        # store chunks into data index
        file_ids = []
        for i in range(0, num_chunks, batch_size):
            if logflag:
                logger.info(f"[ ingest ] Current batch: {i}")
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = batch_chunks

            _, keys = Redis.from_texts_return_keys(
                texts=batch_texts,
                embedding=self.embedder,
                index_name=self.data_index_name,
                redis_url=self.redis_url,
            )
            if logflag:
                logger.info(f"[ ingest ] keys: {keys}")
            file_ids.extend(keys)
            if logflag:
                logger.info(f"[ ingest ] Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")

        # store file_ids into key index
        if not self.check_index_existance(self.key_index_client):
            assert self.create_index(self.key_index_client, self.key_index_name)

        try:
            assert self.store_by_id(self.key_index_client, key=file_name, value="#".join(file_ids))
        except Exception as e:
            if logflag:
                logger.info(f"[ ingest ] {e}. Fail to store chunks of file {file_name}.")
            return False
        return True

    async def check_file_existance(self, file_path: str) -> bool:
        """
        Check whether the file exists in redis database.
        Args:
            file_path (str): The path of the file.
        Returns:
            bool: True if the file exists, False otherwise.
        """
        if logflag:
            logger.info(f"[ check file existance ] file path: {file_path}")
        
        doc_id = "file:" + encode_filename(file_path)
        try:
            key_ids = self.search_by_id(self.key_index_client, doc_id).key_ids
        except Exception as e:
            if logflag:
                logger.info(f"[ check file existance ] {e}. File {file_path} does not exists.")
            return False
        if key_ids:
            if logflag:
                logger.info(f"[ check file existance ] File {file_path} already exists.")
            return True
        return False

    async def get_file_list(self) -> List[dict]:
        """
        Get all ingested file list from redis database.
        Returns:
            List[dict]: The list of file dictionaries.
            [{
                "name": "File Name",
                "id": "File Name",
                "type": "File",
                "parent": "",
            }]
        """

        if logflag:
            logger.info("[ get file list ] start to get file structure")

        # check index existence
        if not self.check_index_existance(self.key_index_client):
            if logflag:
                logger.info(f"[ get file list ] index {self.key_index_name} does not exist")
            return []

        offset = 0
        file_list = []
        while True:
            response = self.client.execute_command("FT.SEARCH", self.key_index_name, "*", "LIMIT", offset, offset + self.search_batch_size)
            # no doc retrieved
            if len(response) < 2:
                break
            file_list = format_search_results(response, file_list)
            offset += self.search_batch_size
            # last batch
            if (len(response) - 1) // 2 < self.search_batch_size:
                break
        if logflag:
            logger.info(f"[ get file list ] final file_list: {file_list}")
        return file_list

    async def get_file_content(self, file_name: str) -> List[dict]:
        """
        Get file content from redis database.
        Args:
            file_name (str): The name of the file.
        Returns:
            List[dict]: The list of file content dictionaries.
        """
        if logflag:
            logger.info(f"[ get file content ] file name: {file_name}")

        # check index existence
        if not self.check_index_existance(self.key_index_client):
            if logflag:
                logger.info(f"[ get file content ] index {self.key_index_client} does not exist")
            return []

        doc_id = "file:" + encode_filename(file_name)
        try:
            file_contents = []
            key_ids = self.search_by_id(self.key_index_client, doc_id).key_ids
            if logflag:
                logger.info(f"[ get file content ] key ids: {key_ids}")
            file_ids = key_ids.split("#")
            for file_id in file_ids:
                content = self.search_by_id(self.data_index_client, file_id)
                if logflag:
                    logger.info(f"[ get file content ] file {file_id} content: {content}")
                file_contents.append(content)
            return file_contents
        except Exception as e:
            if logflag:
                logger.info(f"[ get file content ] {e}. File {file_name} does not exists.")
            return []

    async def delete_all_files(self) -> bool:
        """
        Delete all files in redis database.
        Returns:
            bool: True if all files are deleted successfully, False otherwise.
        """
        if logflag:
            logger.info("[ delete all files ] delete all files")
        
        # drop index KEY_INDEX_NAME
        if self.check_index_existance(self.key_index_client):
            try:
                assert self.drop_index(index_name=self.key_index_name)
            except Exception as e:
                if logflag:
                    logger.error(f"[ delete all files ] {e}. Fail to drop index {self.key_index_name}.")
                return False
        else:
            logger.info(f"[ delete all files ] Index {self.key_index_name} does not exits.")

        # drop index INDEX_NAME
        if self.check_index_existance(self.data_index_client):
            try:
                assert self.drop_index(index_name=self.data_index_name)
            except Exception as e:
                if logflag:
                    logger.error(f"[ delete all files ] {e}. Fail to drop index {self.data_index_name}.")
                return False
        else:
            if logflag:
                logger.info(f"[ delete all files ] Index {self.data_index_name} does not exits.")

        if logflag:
            logger.info("[ delete all files ] successfully delete all files.")
        return True

    async def delete_single_file(self, file_name: str) -> bool:
        """
        Delete single file in redis database.
        Args:
            file_name (str): The name of the file.
        Returns:
            bool: True if the file is deleted successfully, False otherwise.
        """
        if logflag:
            logger.info(f"[ delete single file ] delete file: {file_name}")

        doc_id = "file:" + encode_filename(file_name)
        logger.info(f"[ delete single file ] doc id: {doc_id}")

        # determine whether this file exists in db KEY_INDEX_NAME
        try:
            key_ids = self.search_by_id(self.key_index_client, doc_id).key_ids
        except Exception as e:
            if logflag:
                logger.info(f"[ delete single file ] {e}, File {file_name} does not exists.")
            return True
        file_ids = key_ids.split("#")

        # delete file keys id in db KEY_INDEX_NAME
        try:
            assert self.delete_by_id(self.key_index_client, doc_id)
        except Exception as e:
            if logflag:
                logger.info(f"[ delete single file ] {e}. File {file_name} delete failed for db {KEY_INDEX_NAME}.")
            return False

        # delete file content in db INDEX_NAME
        for file_id in file_ids:
            # determine whether this file exists in db INDEX_NAME
            try:
                self.search_by_id(self.data_index_client, file_id)
            except Exception as e:
                if logflag:
                    logger.info(f"[ delete single file ] {e}. File {file_name} does not exists.")
                return True

            # delete file content
            try:
                assert self.delete_by_id(self.data_index_client, file_id)
            except Exception as e:
                if logflag:
                    logger.info(f"[ delete single file ] {e}. File {file_name} delete failed for db {INDEX_NAME}")
                return False

        return True

    async def similarity_search(self, 
                                input: str, 
                                embedding: list, 
                                search_type: str="similarity", 
                                k: int=4,
                                distance_threshold: Optional[float]=None,
                                score_threshold: Optional[float]=None, 
                                lambda_mult: float=0.2):
        if logflag:
            logger.info(f"[ similarity search ] search type: {search_type}, input: {input}")
        
        if search_type == "similarity":
            search_res = await self.vector_db.asimilarity_search_by_vector(embedding=embedding, k=k)
        elif search_type == "similarity_distance_threshold":
            if distance_threshold is None:
                raise ValueError(
                    "distance_threshold must be provided for " + "similarity_distance_threshold retriever"
                )
            search_res = await self.vector_db.asimilarity_search_by_vector(
                embedding=embedding, k=k, distance_threshold=distance_threshold
            )
        elif search_type == "similarity_score_threshold":
            docs_and_similarities = await self.vector_db.asimilarity_search_with_relevance_scores(
                query=input, k=k, score_threshold=score_threshold
            )
            search_res = [doc for doc, _ in docs_and_similarities]
        elif search_type == "mmr":
            search_res = await self.vector_db.amax_marginal_relevance_search(
                query=input, k=k, fetch_k=k, lambda_mult=lambda_mult
            )
        else:
            if logflag:
                logger.error(f"[ similarity search ] unsupported search type {search_type}")
            raise ValueError(f"{search_type} not valid")

        if logflag:
            logger.info(f"[ similarity search ] search result: {search_res}")
        return search_res

    ############################
    # Redis specific functions #
    ############################
    def create_index(self, client, index_name):
        if logflag:
            logger.info(f"[ create index ] creating index {index_name}")
        try:
            definition = IndexDefinition(index_type=IndexType.HASH, prefix=["file:"])
            client.create_index((TextField("file_name"), TextField("key_ids")), definition=definition)
            if logflag:
                logger.info(f"[ create index ] index {index_name} successfully created")
        except Exception as e:
            if logflag:
                logger.info(f"[ create index ] fail to create index {index_name}: {e}")
            return False
        return True

    def store_by_id(self, client, key, value):
        if logflag:
            logger.info(f"[ store by id ] storing ids of {key}")
        try:
            client.add_document(doc_id="file:" + key, file_name=key, key_ids=value)
            if logflag:
                logger.info(f"[ store by id ] store document success. id: file:{key}")
        except Exception as e:
            if logflag:
                logger.info(f"[ store by id ] fail to store document file:{key}: {e}")
            return False
        return True

    def search_by_id(self, client, doc_id):
        """
        search document by id in redis database.
        """
        if logflag:
            logger.info(f"[ search by id ] searching docs of {doc_id}")
        try:
            results = client.load_document(doc_id)
            if logflag:
                logger.info(f"[ search by id ] search success of {doc_id}: {results}")
            return results
        except Exception as e:
            if logflag:
                logger.info(f"[ search by id ] fail to search docs of {doc_id}: {e}")
            return None

    def check_index_existance(self, client):
        if logflag:
            logger.info(f"[ check index existence ] checking {client}")
        try:
            client.search("*")
            if logflag:
                logger.info(f"[ check index existence ] index of client exists: {client}")
            return True
        except Exception as e:
            if logflag:
                logger.info(f"[ check index existence ] index does not exist: {e}")
            return False

    def drop_index(self, index_name):
        if logflag:
            logger.info(f"[ drop index ] dropping index {index_name}")
        try:
            assert Redis.drop_index(index_name=index_name, delete_documents=True, redis_url=self.redis_url)
            if logflag:
                logger.info(f"[ drop index ] index {index_name} deleted")
        except Exception as e:
            if logflag:
                logger.info(f"[ drop index ] index {index_name} delete failed: {e}")
            return False
        return True

    def delete_by_id(self, client, id):
        try:
            assert client.delete_document(id)
            if logflag:
                logger.info(f"[ delete by id ] delete id success: {id}")
        except Exception as e:
            if logflag:
                logger.info(f"[ delete by id ] fail to delete ids {id}: {e}")
            return False
        return True
