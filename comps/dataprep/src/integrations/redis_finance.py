# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# for test


import json
import os
from pathlib import Path
from typing import List, Optional, Union

import redis
from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain_community.vectorstores import Redis

from comps import OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DataprepRequest
from comps.dataprep.src.integrations.utils.redis_finance_utils import *
from comps.dataprep.src.integrations.utils.redis_kv import RedisKVStore
from comps.dataprep.src.utils import encode_filename, save_content_to_local_disk

logflag = os.getenv("LOGFLAG", False)
upload_folder = "./uploaded_files/"

# Vector Index Configuration
KEY_INDEX_NAME = os.getenv("KEY_INDEX_NAME", "file-keys")
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 600))
SEARCH_BATCH_SIZE = int(os.getenv("SEARCH_BATCH_SIZE", 10))


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


def drop_index(index_name, redis_url=REDIS_URL_VECTOR):
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


def get_all_existing_files():
    file_list = []
    kvstore = RedisKVStore(REDIS_URL_KV)
    file_source_dict = kvstore.get_all("file_source")
    for idx in file_source_dict:
        company_docs = file_source_dict[idx]
        file_list.extend(company_docs["source"])
    return file_list


def check_file_existance(file_name):
    file_list = get_all_existing_files()
    if file_name in file_list:
        return True
    return False


def drop_index_from_kvstore(index_name):
    redis_pool = redis.ConnectionPool.from_url(REDIS_URL_KV)
    client = redis.Redis(connection_pool=redis_pool)
    try:
        client.delete(index_name)
        if logflag:
            logger.info(f"[ drop index ] index {index_name} deleted")
        return True
    except Exception as e:
        if logflag:
            logger.info(f"[ drop index ] index {index_name} delete failed: {e}")
        return False


def drop_record_from_kvstore(index_name, key):
    kvstore = RedisKVStore(REDIS_URL_KV)
    try:
        kvstore.delete(key, index_name)
        if logflag:
            logger.info(f"[ drop record ] record {key} deleted")
        return True
    except Exception as e:
        if logflag:
            logger.info(f"[ drop record ] record {key} delete failed: {e}")
        return False


def remove_company_from_list(company):
    kvstore = RedisKVStore(REDIS_URL_KV)
    company_list = get_company_list()
    try:
        company_list.remove(company)
        kvstore.put("company", {"company": company_list}, "company_list")
        if logflag:
            logger.info(f"[ remove company ] company {company} removed from company list")
        return True
    except Exception as e:
        if logflag:
            logger.info(f"[ remove company ] company {company} remove failed: {e}")
        return False


async def ingest_financial_data(filename: str):
    """
    1 vector store - multiple collections: chunks/tables (embeddings for summaries), doc_titles
    1 kv store - multiple collections: full doc, chunks, tables
    """
    file_ids = []

    conv_res, full_doc, metadata = parse_doc_and_extract_metadata(filename)
    if not metadata:
        raise HTTPException(status_code=400, detail="Failed to extract metadata from the document.")

    if not filename.endswith(".pdf"):
        full_doc = post_process_html(full_doc, metadata["doc_title"])

    # save company name
    metadata = save_company_name(metadata)

    # save file source info and check if file already exists
    file_existed = save_file_source(filename, metadata)
    if file_existed:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file {filename} already exists. Please upload a different file.",
        )

    # save full doc
    save_full_doc(full_doc, metadata)

    # save doc_title
    doc_title = metadata["doc_title"]
    keys = save_doc_title(doc_title, metadata)
    file_ids.extend(keys)

    # chunk and save
    keys = split_markdown_and_summarize_save(full_doc, metadata)
    file_ids.extend(keys)

    # process tables and save
    keys = process_tables(conv_res, metadata)
    file_ids.extend(keys)
    # save_file_ids_to_filekey_index(doc_id, file_ids)


@OpeaComponentRegistry.register("OPEA_DATAPREP_REDIS_FINANCE")
class OpeaRedisDataprepFinance(OpeaComponent):
    """A specialized dataprep component derived from OpeaComponent for redis dataprep services.

    Attributes:
        client (redis.Redis): An instance of the redis client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.db_client = self._initialize_client(REDIS_URL_VECTOR)
        self.kv_client = self._initialize_client(REDIS_URL_KV)
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaRedisDataprepFinance health check failed.")

    def _initialize_client(self, url) -> redis.Redis:
        if logflag:
            logger.info("[ initialize client ] initializing redis client...")

        """Initializes the redis client."""
        try:
            redis_pool = redis.ConnectionPool.from_url(url)
            client = redis.Redis(connection_pool=redis_pool)
            return client
        except Exception as e:
            logger.error(f"fail to initialize redis client: {e}")
            return None

    def check_health(self) -> bool:
        """Checks the health of the dataprep service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ health check ] start to check health of redis")
        try:
            if self.db_client.ping():
                if logflag:
                    logger.info("[ health check ] Successfully connected to Redis vector db!")
            if self.kv_client.ping():
                if logflag:
                    logger.info("[ health check ] Successfully connected to Redis kv store!")
            return True
        except redis.ConnectionError as e:
            logger.info(f"[ health check ] Failed to connect to Redis: {e}")
            return False

    def invoke(self, *args, **kwargs):
        pass

    async def ingest_files(
        self,
        input: DataprepRequest,
    ):
        """Ingest files/links content into redis database.

        Save in the format of vector[768].
        Returns '{"status": 200, "message": "Data preparation succeeded"}' if successful.
        Args:
            input (DataprepRequest): Model containing the following parameters:
                files (Union[UploadFile, List[UploadFile]], optional): A file or a list of files to be ingested. Defaults to File(None).
                link_list (str, optional): A list of links to be ingested. Defaults to Form(None).
                chunk_size (int, optional): The size of the chunks to be split. Defaults to Form(1500).
                chunk_overlap (int, optional): The overlap between chunks. Defaults to Form(100).
                process_table (bool, optional): Whether to process tables in PDFs. Defaults to Form(False).
                table_strategy (str, optional): The strategy to process tables in PDFs. Defaults to Form("fast").
        """
        files = input.files
        link_list = input.link_list

        if logflag:
            logger.info(f"[ redis ingest ] files:{files}")
            logger.info(f"[ redis ingest ] link_list:{link_list}")

        if files:
            if not isinstance(files, list):
                files = [files]
            uploaded_files = []

            for file in files:
                if not file.filename.lower().endswith(".pdf"):
                    raise HTTPException(status_code=400, detail="Only PDF files are supported.")

                if check_file_existance(file.filename):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded file {file.filename} already exists. Please upload a different file.",
                    )

                encode_file = encode_filename(file.filename)
                save_path = upload_folder + encode_file
                await save_content_to_local_disk(save_path, file)
                await ingest_financial_data(save_path)
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
                if not link.startswith("http"):
                    raise HTTPException(status_code=400, detail=f"Link {link} is not a valid URL.")
                if check_file_existance(link):
                    raise HTTPException(
                        status_code=400, detail=f"Uploaded link {link} already exists. Please upload a different link."
                    )
                if logflag:
                    logger.info(f"[ redis ingest] processing link {link}")

                await ingest_financial_data(link)
            if logflag:
                logger.info(f"[ redis ingest] Successfully saved link list {link_list}")
            return {"status": 200, "message": "Data preparation succeeded"}

        raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

    async def get_files(self):
        """Get file source names."""

        if logflag:
            logger.info("[ redis get ] start to get filenames of all uploaded files")

        try:
            file_list = get_all_existing_files()
            if logflag:
                logger.info(f"[ redis get ] Successfully get files: {file_list}")
        except:
            if logflag:
                logger.info("[ redis get ] Fail to get files.")
            raise HTTPException(status_code=500, detail="Fail to get files.")
        return file_list

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file related to `file_path` - company name.

        `file_path`:
            - specific company name: delete all files related to the company
        """
        if logflag:
            logger.info(f"[ redis delete ] delete files related to: {file_path}")

        company_list = get_company_list()
        company = file_path.upper()
        if company not in company_list:
            if logflag:
                logger.info(f"[ redis delete ] Company {file_path} does not exists.")
            raise HTTPException(
                status_code=404,
                detail=f"Company {file_path} does not exists. Please choose from the following list {company_list}.",
            )

        # delete files related to the company
        # delete chunks_{company} and tables_{company} from vector store
        # delete full_doc_{company} from kv store
        # delete doc_titles_{company} from kv store
        # delete {company} from file_source in kv store
        try:
            drop_index(index_name=f"chunks_{company}")
            drop_index(index_name=f"tables_{company}")
            drop_index(index_name=f"titles_{company}")
            drop_index_from_kvstore(index_name=f"chunks_{company}")
            drop_index_from_kvstore(index_name=f"tables_{company}")
            drop_index_from_kvstore(index_name=f"full_doc_{company}")
            drop_record_from_kvstore(index_name="file_source", key=company)
            remove_company_from_list(company)

            if logflag:
                logger.info(f"[ redis delete ] Successfully delete files related to company {file_path}")
            return {"status": True}
        except:
            if logflag:
                logger.info(f"[ redis delete ] Fail to delete files related to company {file_path}.")
            raise HTTPException(status_code=500, detail=f"Fail to delete files related to company {file_path}.")
