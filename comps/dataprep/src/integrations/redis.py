# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# for test


import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional, Union

import redis
from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Redis
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter
from redis import asyncio as aioredis
from redis.commands.search.field import TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DataprepRequest, RedisDataprepRequest
from comps.dataprep.src.utils import (
    create_upload_folder,
    document_loader,
    encode_filename,
    format_search_results,
    get_separators,
    get_tables_result,
    parse_html_new,
    remove_folder_with_ignore,
    save_content_to_local_disk,
)

logger = CustomLogger("redis_dataprep")
logflag = os.getenv("LOGFLAG", False)
upload_folder = "./uploaded_files/"


# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# TEI Embedding endpoints
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
# Huggingface API token for TEI embedding endpoint
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Vector Index Configuration
INDEX_NAME = os.getenv("INDEX_NAME", "rag_redis")
KEY_INDEX_NAME = os.getenv("KEY_INDEX_NAME", "file-keys")
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 600))
SEARCH_BATCH_SIZE = int(os.getenv("SEARCH_BATCH_SIZE", 10))

# Redis Connection Information
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


def get_boolean_env_var(var_name, default_value=False):
    """Retrieve the boolean value of an environment variable.

    Args:
    var_name (str): The name of the environment variable to retrieve.
    default_value (bool): The default value to return if the variable
    is not found.

    Returns:
    bool: The value of the environment variable, interpreted as a boolean.
    """
    true_values = {"true", "1", "t", "y", "yes"}
    false_values = {"false", "0", "f", "n", "no"}

    # Retrieve the environment variable's value
    value = os.getenv(var_name, "").lower()

    # Decide the boolean value based on the content of the string
    if value in true_values:
        return True
    elif value in false_values:
        return False
    else:
        return default_value


def format_redis_conn_from_env():
    redis_url = os.getenv("REDIS_URL", None)
    if redis_url:
        return redis_url
    else:
        using_ssl = get_boolean_env_var("REDIS_SSL", False)
        start = "rediss://" if using_ssl else "redis://"

        # if using RBAC
        password = os.getenv("REDIS_PASSWORD", None)
        username = os.getenv("REDIS_USERNAME", "default")
        if password is not None:
            start += f"{username}:{password}@"

        return start + f"{REDIS_HOST}:{REDIS_PORT}"


REDIS_URL = format_redis_conn_from_env()
redis_pool = redis.ConnectionPool.from_url(REDIS_URL)


async def check_index_existance(client):
    if logflag:
        logger.info(f"[ check index existence ] checking {client}")
    try:
        results = await client.search("*")
        if logflag:
            logger.info(f"[ check index existence ] index of client exists: {client}")
        return results
    except Exception as e:
        if logflag:
            logger.info(f"[ check index existence ] index does not exist: {e}")
        return None


async def create_index(client, index_name: str = KEY_INDEX_NAME):
    if logflag:
        logger.info(f"[ create index ] creating index {index_name}")
    try:
        definition = IndexDefinition(index_type=IndexType.HASH, prefix=["file:"])
        await client.create_index(
            (TextField("file_name"), TextField("key_ids"), TextField("index_name")), definition=definition
        )
        if logflag:
            logger.info(f"[ create index ] index {index_name} successfully created")
    except Exception as e:
        if logflag:
            logger.info(f"[ create index ] fail to create index {index_name}: {e}")
        return False
    return True


async def store_by_id(client, key, value, ingest_index_name):
    if logflag:
        logger.info(f"[ store by id ] storing ids of {key}")
    try:
        await client.add_document(doc_id="file:" + key, file_name=key, key_ids=value, index_name=ingest_index_name)
        if logflag:
            logger.info(f"[ store by id ] store document success. id: file:{key}")
    except Exception as e:
        if logflag:
            logger.info(f"[ store by id ] fail to store document file:{key}: {e}")
        return False
    return True


async def search_by_id(client, doc_id):
    if logflag:
        logger.info(f"[ search by id ] searching docs of {doc_id}")
    try:
        results = await client.load_document(doc_id)
        if logflag:
            logger.info(f"[ search by id ] search success of {doc_id}: {results}")
        return results
    except Exception as e:
        if logflag:
            logger.info(f"[ search by id ] fail to search docs of {doc_id}: {e}")
        return None


def drop_index(index_name, redis_url=REDIS_URL):
    if logflag:
        logger.info(f"[ drop index ] dropping index {index_name}")
    try:
        assert Redis.drop_index(index_name=index_name, delete_documents=True, redis_url=redis_url)
        if logflag:
            logger.info(f"[ drop index ] index {index_name} deleted")
    except Exception as e:
        if logflag:
            logger.info(f"[ drop index ] index {index_name} delete failed: {e}")
        return False
    return True


async def delete_by_id(client, id):
    try:
        res = await client.delete_document(id)
        assert res
        if logflag:
            logger.info(f"[ delete by id ] delete id success: {id}")
    except Exception as e:
        if logflag:
            logger.info(f"[ delete by id ] fail to delete ids {id}: {e}")
        return False
    return True


async def ingest_chunks_to_redis(file_name: str, chunks: List, embedder, index_name: str):
    if logflag:
        logger.info(f"[ redis ingest chunks ] file name: {file_name}")

    # Batch size
    batch_size = 32
    num_chunks = len(chunks)

    # if data will be saved to a different index name than the default one
    ingest_index_name = index_name if index_name else INDEX_NAME

    file_ids = []
    for i in range(0, num_chunks, batch_size):
        if logflag:
            logger.info(f"[ redis ingest chunks ] Current batch: {i}")
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = batch_chunks

        _, keys = await asyncio.to_thread(
            Redis.from_texts_return_keys,
            texts=batch_texts,
            embedding=embedder,
            index_name=ingest_index_name,
            redis_url=REDIS_URL,
        )
        if logflag:
            logger.info(f"[ redis ingest chunks ] keys: {keys}")
        file_ids.extend(keys)
        if logflag:
            logger.info(f"[ redis ingest chunks ] Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")

    # store file_ids into index file-keys
    r = await aioredis.from_url(REDIS_URL)
    client = r.ft(KEY_INDEX_NAME)
    if not await check_index_existance(client):
        await create_index(client)

    try:
        await store_by_id(
            client,
            key=encode_filename(ingest_index_name) + "_" + file_name,
            value="#".join(file_ids),
            ingest_index_name=ingest_index_name,
        )
    except Exception as e:
        if logflag:
            logger.info(f"[ redis ingest chunks ] {e}. Fail to store chunks of file {file_name}.")
        raise HTTPException(status_code=500, detail=f"Fail to store chunks of file {file_name}.")
    return True


async def ingest_data_to_redis(doc_path: DocPath, embedder, index_name):
    """Ingest document to Redis."""
    path = doc_path.path
    if logflag:
        logger.info(f"[ redis ingest data ] Parsing document {path}.")

    if path.endswith(".html"):
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ]
        text_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=doc_path.chunk_size,
            chunk_overlap=doc_path.chunk_overlap,
            add_start_index=True,
            separators=get_separators(),
        )

    content = await document_loader(path)
    if logflag:
        logger.info("[ redis ingest data ] file content loaded")

    structured_types = [".xlsx", ".csv", ".json", "jsonl"]
    _, ext = os.path.splitext(path)

    if ext in structured_types:
        chunks = content
    else:
        chunks = await asyncio.to_thread(text_splitter.split_text, content)

    ### Specially processing for the table content in PDFs
    if doc_path.process_table and path.endswith(".pdf"):
        table_chunks = get_tables_result(path, doc_path.table_strategy)
        logger.info(f"[ redis ingest data ] table chunks: {table_chunks}")
        if table_chunks:
            chunks = chunks + table_chunks
        else:
            logger.info(f"[ redis ingest data ] No table chunks found in {path}.")
    if logflag:
        logger.info(f"[ redis ingest data ] Done preprocessing. Created {len(chunks)} chunks of the given file.")

    file_name = doc_path.path.split("/")[-1]
    return await ingest_chunks_to_redis(file_name, chunks, embedder, index_name)


@OpeaComponentRegistry.register("OPEA_DATAPREP_REDIS")
class OpeaRedisDataprep(OpeaComponent):
    """A specialized dataprep component derived from OpeaComponent for redis dataprep services.

    Attributes:
        client (redis.Redis): An instance of the redis client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.client = redis.Redis(connection_pool=redis_pool)
        self.data_index_client, self.key_index_client = asyncio.run(self._initialize_client())
        self.embedder = asyncio.run(self._initialize_embedder())
        health_status = asyncio.run(self.check_health())
        if not health_status:
            logger.error("OpeaRedisDataprep health check failed.")

    async def _initialize_client(self) -> redis.Redis:
        if logflag:
            logger.info("[ initialize client ] initializing redis client...")

        """Initializes the redis client."""
        try:
            client = await aioredis.from_url(REDIS_URL)
            data_index_client = client.ft(INDEX_NAME)
            key_index_client = client.ft(KEY_INDEX_NAME)
            return data_index_client, key_index_client
        except Exception as e:
            logger.error(f"fail to initialize redis client: {e}")
            return None

    async def _initialize_embedder(self):
        if TEI_EMBEDDING_ENDPOINT:
            if not HUGGINGFACEHUB_API_TOKEN:
                raise HTTPException(
                    status_code=400,
                    detail="You MUST offer the `HUGGINGFACEHUB_API_TOKEN` when using `TEI_EMBEDDING_ENDPOINT`.",
                )

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(TEI_EMBEDDING_ENDPOINT + "/info")
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400, detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available."
                    )
                model_id = response.json()["model_id"]
            # create embeddings using TEI endpoint service
            embedder = HuggingFaceInferenceAPIEmbeddings(
                api_key=HUGGINGFACEHUB_API_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        else:
            # create embeddings using local embedding model
            embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return embedder

    async def check_health(self) -> bool:
        """Checks the health of the dataprep service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ health check ] start to check health of redis")
        try:
            if self.client.ping():
                if logflag:
                    logger.info("[ health check ] Successfully connected to Redis!")
                return True
        except redis.ConnectionError as e:
            logger.info(f"[ health check ] Failed to connect to Redis: {e}")
            return False

    def invoke(self, *args, **kwargs):
        pass

    async def ingest_files(self, input: Union[DataprepRequest, RedisDataprepRequest]):
        """Ingest files/links content into redis database.

        Save in the format of vector[768].
        Returns '{"status": 200, "message": "Data preparation succeeded"}' if successful.
        Args:
            input (RedisDataprepRequest): Model containing the following parameters:
                files (Union[UploadFile, List[UploadFile]], optional): A file or a list of files to be ingested. Defaults to File(None).
                link_list (str, optional): A list of links to be ingested. Defaults to Form(None).
                chunk_size (int, optional): The size of the chunks to be split. Defaults to Form(1500).
                chunk_overlap (int, optional): The overlap between chunks. Defaults to Form(100).
                process_table (bool, optional): Whether to process tables in PDFs. Defaults to Form(False).
                table_strategy (str, optional): The strategy to process tables in PDFs. Defaults to Form("fast").
                index_name (str, optional): The name of the index where data will be ingested.
        """
        files = input.files
        link_list = input.link_list
        chunk_size = input.chunk_size
        chunk_overlap = input.chunk_overlap
        process_table = input.process_table
        table_strategy = input.table_strategy

        index_name = None
        if isinstance(input, RedisDataprepRequest):
            index_name = input.index_name

        if logflag:
            logger.info(f"[ redis ingest ] files:{files}")
            logger.info(f"[ redis ingest ] link_list:{link_list}")

        if index_name and index_name.lower() == "all":
            raise HTTPException(
                status_code=400,
                detail="'all' cannot be used as an index_name because it is reserved for specific functionality within the service.",
            )

        if files:
            if not isinstance(files, list):
                files = [files]
            uploaded_files = []

            for file in files:
                encode_file = encode_filename(file.filename)
                index_name_id = encode_filename(INDEX_NAME if index_name is None else index_name)
                doc_id = "file:" + index_name_id + "_" + encode_file

                if logflag:
                    logger.info(f"[ redis ingest ] processing file {doc_id}")

                # check whether the file already exists
                key_ids = None
                try:
                    result = await search_by_id(self.key_index_client, doc_id)
                    key_ids = result.key_ids
                    if logflag:
                        logger.info(f"[ redis ingest] File {file.filename} already exists.")
                except Exception as e:
                    logger.info(f"[ redis ingest] File {file.filename} does not exist.")
                if key_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded file {file.filename} already exists. Please change file name or index name.",
                    )

                save_path = upload_folder + encode_file
                await save_content_to_local_disk(save_path, file)
                await ingest_data_to_redis(
                    DocPath(
                        path=save_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        process_table=process_table,
                        table_strategy=table_strategy,
                    ),
                    self.embedder,
                    index_name,
                )
                uploaded_files.append(save_path)
                if logflag:
                    logger.info(f"[ redis ingest] Successfully saved file {save_path}")

            result = {"status": 200, "message": "Data preparation succeeded"}
            if logflag:
                logger.info(result)
            return result

        if link_list:
            link_list = json.loads(link_list)  # Parse JSON string to list
            if not isinstance(link_list, list):
                raise HTTPException(status_code=400, detail=f"Link_list {link_list} should be a list.")
            for link in link_list:
                encoded_link = encode_filename(link)
                index_name_id = encode_filename(INDEX_NAME if index_name is None else index_name)
                doc_id = "file:" + index_name_id + "_" + encoded_link + ".txt"
                if logflag:
                    logger.info(f"[ redis ingest] processing link {doc_id}")

                # check whether the link file already exists
                key_ids = None
                try:
                    result = await search_by_id(self.key_index_client, doc_id)
                    key_ids = result.key_ids
                    if logflag:
                        logger.info(f"[ redis ingest] Link {link} already exists.")
                except Exception as e:
                    logger.info(f"[ redis ingest] Link {link} does not exist. Keep storing.")
                if key_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded link {link} already exists. Please change another link or index_name.",
                    )

                save_path = upload_folder + encoded_link + ".txt"
                content = parse_html_new([link], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                await save_content_to_local_disk(save_path, content)
                await ingest_data_to_redis(
                    DocPath(
                        path=save_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        process_table=process_table,
                        table_strategy=table_strategy,
                    ),
                    self.embedder,
                    index_name,
                )
            if logflag:
                logger.info(f"[ redis ingest] Successfully saved link list {link_list}")
            return {"status": 200, "message": "Data preparation succeeded"}

        raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

    async def get_files(self, index_name: str = Body(None, embed=True)):
        """Get file structure from redis database in the format of
        {
            "name": "File Name",
            "id": "File Name",
            "type": "File",
            "parent": "",
        }"""

        if logflag:
            logger.info("[ redis get ] start to get file structure")

        offset = 0
        file_list = []

        # check index existence
        res = await check_index_existance(self.key_index_client)
        if not res:
            if logflag:
                logger.info(f"[ redis get ] index {KEY_INDEX_NAME} does not exist")
            return file_list

        while True:
            response = self.client.execute_command(
                "FT.SEARCH", KEY_INDEX_NAME, "*", "LIMIT", offset, offset + SEARCH_BATCH_SIZE
            )
            # no doc retrieved
            if len(response) < 2:
                break

            file_list = format_search_results(response, file_list)
            index_name = INDEX_NAME if index_name is None else index_name
            filtered_files = []
            for file in file_list:
                # Ensure "index_name" key exists in the file dictionary
                if "index_name" not in file:
                    continue

                # Check if the file should be included based on the index_name
                if index_name == "all" or index_name == file["index_name"]:
                    # Remove the index_name prefix from "name" and "id"
                    prefix = f"{file['index_name']}_"
                    file["name"] = file["name"].replace(prefix, "", 1)
                    file["id"] = file["id"].replace(prefix, "", 1)
                    filtered_files.append(file)

            file_list = filtered_files
            offset += SEARCH_BATCH_SIZE
            # last batch
            if (len(response) - 1) // 2 < SEARCH_BATCH_SIZE:
                break
        if logflag:
            logger.info(f"[get] final file_list: {file_list}")
        return file_list

    async def delete_files(self, file_path: str = Body(..., embed=True), index_name: str = Body(None, embed=True)):
        """Delete file according to `file_path`.

        `file_path`:
            - specific file path (e.g. /path/to/file.txt)
            - "all": delete all files uploaded
        """
        if logflag:
            logger.info(f"[ redis delete ] delete files: {file_path}")

        # delete all uploaded files
        if file_path == "all" and index_name is None:
            if logflag:
                logger.info("[ redis delete ] delete all files")

            # drop index KEY_INDEX_NAME
            if await check_index_existance(self.key_index_client):
                try:
                    assert drop_index(index_name=KEY_INDEX_NAME)
                except Exception as e:
                    if logflag:
                        logger.info(f"[ redis delete ] {e}. Fail to drop index {KEY_INDEX_NAME}.")
                    raise HTTPException(status_code=500, detail=f"Fail to drop index {KEY_INDEX_NAME}.")
            else:
                logger.info(f"[ redis delete ] Index {KEY_INDEX_NAME} does not exits.")

            if len(self.get_list_of_indices()) > 0:
                for i in self.get_list_of_indices():
                    try:
                        # drop index INDEX_NAME
                        assert drop_index(index_name=i)
                        logger.info(f"[ redis delete ]  Index_name: {i} is deleted.")
                    except Exception as e:
                        if logflag:
                            logger.info(f"[ redis delete ] {e}. Fail to drop index {i}.")
                        raise HTTPException(status_code=500, detail=f"Fail to drop index {i}.")
            else:
                if logflag:
                    logger.info("[ redis delete ] There is no index_name registered to redis db.")

            # delete files on local disk
            try:
                remove_folder_with_ignore(upload_folder)
            except Exception as e:
                if logflag:
                    logger.info(f"[ redis delete ] {e}. Fail to delete {upload_folder}.")
                raise HTTPException(status_code=500, detail=f"Fail to delete {upload_folder}.")

            if logflag:
                logger.info("[ redis delete ] successfully delete all files.")
            create_upload_folder(upload_folder)
            if logflag:
                logger.info({"status": True})
            return {"status": True}

        # partially delete files based on index_name
        index_name = INDEX_NAME if index_name is None else index_name
        index_name_id = encode_filename(index_name)

        if file_path == "all" and index_name is not None:
            file_list = [i["name"] for i in await self.get_files(index_name)]
        else:
            file_list = [file_path]

        for file_path in file_list:
            delete_path = Path(upload_folder + "/" + encode_filename(file_path))
            if logflag:
                logger.info(f"[ redis delete ] delete_path: {delete_path}")

            encode_file = encode_filename(file_path)
            doc_id = "file:" + index_name_id + "_" + encode_file
            logger.info(f"[ redis delete ] doc id: {doc_id}")

            # determine whether this file exists in db KEY_INDEX_NAME
            try:
                result = await search_by_id(self.key_index_client, doc_id)
                key_ids = result.key_ids
            except Exception as e:
                if logflag:
                    logger.info(f"[ redis delete ] {e}, File {file_path} does not exists.")
                raise HTTPException(
                    status_code=404, detail=f"File not found in db {KEY_INDEX_NAME}. Please check file_path."
                )
            file_ids = key_ids.split("#")

            # delete file keys id in db KEY_INDEX_NAME
            try:
                res = await delete_by_id(self.key_index_client, doc_id)
                assert res
            except Exception as e:
                if logflag:
                    logger.info(f"[ redis delete ] {e}. File {file_path} delete failed for db {KEY_INDEX_NAME}.")
                raise HTTPException(status_code=500, detail=f"File {file_path} delete failed for key index.")

            # delete file content in db index_name
            for file_id in file_ids:
                # determine whether this file exists in db index_name
                try:
                    await search_by_id(self.data_index_client, file_id)
                except Exception as e:
                    if logflag:
                        logger.info(f"[ redis delete ] {e}. File {file_path} does not exists.")
                    raise HTTPException(
                        status_code=404, detail=f"File not found in db {index_name}. Please check file_path."
                    )

                # delete file content
                try:
                    res = await delete_by_id(self.data_index_client, file_id)
                    assert res
                except Exception as e:
                    if logflag:
                        logger.info(f"[ redis delete ] {e}. File {file_path} delete failed for db {index_name}")
                    raise HTTPException(status_code=500, detail=f"File {file_path} delete failed for index.")

            # local file does not exist (restarted docker container)
            if not delete_path.exists():
                if logflag:
                    logger.info(f"[ redis delete ] File {file_path} not saved locally.")
                return {"status": True}

            # delete local file
            if delete_path.is_file():
                # delete file on local disk
                delete_path.unlink()
                if logflag:
                    logger.info(f"[ redis delete ] File {file_path} deleted successfully.")

            # delete folder
            else:
                if logflag:
                    logger.info(f"[ redis delete ] Delete folder {file_path} is not supported for now.")
                raise HTTPException(status_code=404, detail=f"Delete folder {file_path} is not supported for now.")

        return {"status": True}

    def get_list_of_indices(self):
        """Retrieves a list of all indices from the Redis client.

        Returns:
            A list of index names as strings.
        """
        # Execute the command to list all indices
        indices = self.client.execute_command("FT._LIST")
        # Decode each index name from bytes to string
        indices_list = [item.decode("utf-8") for item in indices]
        return indices_list
