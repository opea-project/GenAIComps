# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import List, Optional, Union

from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter
from opensearchpy import OpenSearch

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DataprepRequest
from comps.dataprep.src.utils import (
    create_upload_folder,
    document_loader,
    encode_filename,
    get_separators,
    get_tables_result,
    parse_html,
    remove_folder_with_ignore,
    save_content_to_local_disk,
)

logger = CustomLogger("opea_dataprep_opensearch")
logflag = os.getenv("LOGFLAG", False)


class Config:
    """Configuration class to store environment variables and default settings."""

    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
    OPENSEARCH_INITIAL_ADMIN_PASSWORD = os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD", "")
    OPENSEARCH_SSL = os.getenv("OPENSEARCH_SSL", "false").lower() == "true"
    OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", None)
    INDEX_NAME = os.getenv("INDEX_NAME", "rag-opensearch")
    KEY_INDEX_NAME = os.getenv("KEY_INDEX_NAME", "file-keys")
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 600))
    SEARCH_BATCH_SIZE = int(os.getenv("SEARCH_BATCH_SIZE", 10))

    @staticmethod
    def get_boolean_env_var(var_name, default_value=False):
        """Retrieve the boolean value of an environment variable."""
        true_values = {"true", "1", "t", "y", "yes"}
        false_values = {"false", "0", "f", "n", "no"}
        value = os.getenv(var_name, "").lower()
        if value in true_values:
            return True
        elif value in false_values:
            return False
        else:
            return default_value

    @staticmethod
    def format_opensearch_conn_from_env():
        """Format the OpenSearch connection URL based on environment variables."""
        opensearch_url = Config.OPENSEARCH_URL
        if opensearch_url:
            return opensearch_url
        else:
            start = "https://" if Config.OPENSEARCH_SSL else "http://"
            return f"{start}{Config.OPENSEARCH_HOST}:{Config.OPENSEARCH_PORT}"


# Initialize the OpenSearch URL based on configuration
OPENSEARCH_URL = Config.format_opensearch_conn_from_env()


@OpeaComponentRegistry.register("OPEA_DATAPREP_OPENSEARCH")
class OpeaOpenSearchDataprep(OpeaComponent):
    """Dataprep component for OpenSearch ingestion and search services."""

    def __init__(self, name: str, description: str, config: dict = None):
        self.upload_folder = "./uploaded_files/"
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        # Initialize embeddings
        TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
        if TEI_EMBEDDING_ENDPOINT:
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
            self.embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=HUGGINGFACEHUB_API_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBED_MODEL)

        # OpenSearch client setup
        self.auth = ("admin", Config.OPENSEARCH_INITIAL_ADMIN_PASSWORD)
        self.opensearch_client = OpenSearchVectorSearch(
            opensearch_url=OPENSEARCH_URL,
            index_name=Config.INDEX_NAME,
            # Default engine for OpenSearch is "nmslib",
            # but "nmslib" engine is deprecated in OpenSearch and cannot be used for new index creation in OpenSearch from 3.0.0.
            engine="faiss",
            embedding_function=self.embeddings,
            http_auth=self.auth,
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )

        # Perform health check
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaOpenSearchDataprep health check failed.")

    def check_health(self) -> bool:
        """Checks the health of the OpenSearch service."""
        try:
            client = OpenSearch(
                hosts=[{"host": Config.OPENSEARCH_HOST, "port": Config.OPENSEARCH_PORT}],
                http_auth=self.auth,
                use_ssl=True,
                verify_certs=False,
            )
            info = client.info()
            logger.info(f"[ health check ] OpenSearch info: {info}")
            return True
        except Exception as e:
            logger.error(f"[ health check ] Failed to connect to OpenSearch: {e}")
            return False

    def invoke(self, *args, **kwargs):
        pass

    def check_index_existence(self, client, index_name):
        """Check if an index exists in OpenSearch."""
        try:
            exists = client.index_exists(index_name) or False
            if exists:
                logger.info(f"[ check index existence ] Index {index_name} exists.")
            else:
                logger.info(f"[ check index existence ] Index {index_name} does not exist.")
            return exists
        except Exception as e:
            logger.error(f"[ check index existence ] Error checking index {index_name}: {e}")
            return False

    def create_index(self, client, index_name: str = Config.KEY_INDEX_NAME):
        """Create a new index in OpenSearch."""
        try:
            index_body = {
                "mappings": {
                    "properties": {
                        "file_name": {"type": "text"},
                        "key_ids": {"type": "text"},
                    }
                }
            }
            client.client.indices.create(index_name, body=index_body)
            logger.info(f"[ create index ] Index {index_name} created successfully.")
            return True
        except Exception as e:
            logger.error(f"[ create index ] Failed to create index {index_name}: {e}")
            return False

    def store_by_id(self, client, key, value):
        if logflag:
            logger.info(f"[ store by id ] storing ids of {key}")
        try:
            client.client.index(
                index=Config.KEY_INDEX_NAME,
                body={"file_name": f"file:${key}", "key_ids:": value},
                id="file:" + key,
                refresh=True,
            )
            if logflag:
                logger.info(f"[ store by id ] store document success. id: file:{key}")
        except Exception as e:
            if logflag:
                logger.info(f"[ store by id ] fail to store document file:{key}: {e}")
            return False
        return True

    def search_by_id(self, client, doc_id):
        if logflag:
            logger.info(f"[ search by id ] searching docs of {doc_id}")
        try:
            result = client.client.get(index=Config.KEY_INDEX_NAME, id=doc_id)
            if result["found"]:
                if logflag:
                    logger.info(f"[ search by id ] search success of {doc_id}: {result}")
                return result
            return None
        except Exception as e:
            if logflag:
                logger.info(f"[ search by id ] fail to search docs of {doc_id}: {e}")
            return None

    def drop_index(self, client: OpenSearchVectorSearch, index_name: str):
        if logflag:
            logger.info(f"[ drop index ] dropping index {index_name}")
        try:
            client.client.indices.delete(index=index_name)
            if logflag:
                logger.info(f"[ drop index ] index {index_name} deleted")
        except Exception as e:
            if logflag:
                logger.info(f"[ drop index ] index {index_name} delete failed: {e}")
            return False
        return True

    def delete_by_id(self, client, doc_id):
        try:
            response = client.client.delete(index=Config.KEY_INDEX_NAME, id=doc_id)
            if response["result"] == "deleted":
                if logflag:
                    logger.info(f"[ delete by id ] delete id success: {doc_id}")
                return True
            else:
                if logflag:
                    logger.info(f"[ delete by id ] delete id failed: {doc_id}")
                return False
        except Exception as e:
            if logflag:
                logger.info(f"[ delete by id ] fail to delete ids {doc_id}: {e}")
            return False

    def ingest_chunks_to_opensearch(self, file_name: str, chunks: List):
        """Ingest chunks of text data to OpenSearch."""
        batch_size = 32
        num_chunks = len(chunks)
        file_ids = []

        for i in range(0, num_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            keys = self.opensearch_client.add_texts(
                texts=batch_chunks, metadatas=[{"source": file_name}] * len(batch_chunks)
            )
            file_ids.extend(keys)
            logger.info(f"[ ingest chunks ] Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")

        if not self.check_index_existence(self.opensearch_client, Config.KEY_INDEX_NAME):
            self.create_index(self.opensearch_client)

        try:
            self.store_by_id(self.opensearch_client, key=file_name, value="#".join(file_ids))
        except Exception as e:
            logger.error(f"[ ingest chunks ] Failed to store chunks of file {file_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to store chunks of file {file_name}.")
        return True

    async def ingest_data_to_opensearch(self, doc_path: DocPath):
        """Ingest document to OpenSearch."""
        path = doc_path.path
        if logflag:
            logger.info(f"[ ingest data ] Parsing document {path}.")

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
            logger.info("[ ingest data ] file content loaded")

        structured_types = [".xlsx", ".csv", ".json", "jsonl"]
        _, ext = os.path.splitext(path)

        if ext in structured_types:
            chunks = content
        else:
            chunks = text_splitter.split_text(content)

        ### Specially processing for the table content in PDFs
        if doc_path.process_table and path.endswith(".pdf"):
            table_chunks = get_tables_result(path, doc_path.table_strategy)
            logger.info(f"[ ingest data ] table chunks: {table_chunks}")
            if table_chunks:
                chunks = chunks + table_chunks
            else:
                logger.info(f"[ ingest data ] No table chunks found in {path}.")
        if logflag:
            logger.info(f"[ ingest data ] Done preprocessing. Created {len(chunks)} chunks of the given file.")

        file_name = doc_path.path.split("/")[-1]
        return self.ingest_chunks_to_opensearch(file_name, chunks)

    def search_all_documents(self, index_name, offset, search_batch_size):
        try:
            response = self.opensearch_client.client.search(
                index=index_name,
                body={
                    "query": {"match_all": {}},
                    "from": offset,  # Starting position
                    "size": search_batch_size,  # Number of results to return
                },
            )
            # Get total number of matching documents
            total_hits = response["hits"]["total"]["value"]
            # Get the documents from the current batch
            documents = response["hits"]["hits"]

            return {"total_hits": total_hits, "documents": documents}

        except Exception as e:
            print(f"Error performing search: {e}")
            return None

    async def ingest_files(
        self,
        input: DataprepRequest,
    ):
        """Ingest files/links content into opensearch database.

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
        chunk_size = input.chunk_size
        chunk_overlap = input.chunk_overlap
        process_table = input.process_table
        table_strategy = input.table_strategy

        if logflag:
            logger.info(f"[ upload ] files:{files}")
            logger.info(f"[ upload ] link_list:{link_list}")

        if files:
            if not isinstance(files, list):
                files = [files]
            uploaded_files = []

            for file in files:
                encode_file = encode_filename(file.filename)
                doc_id = "file:" + encode_file
                if logflag:
                    logger.info(f"[ upload ] processing file {doc_id}")

                # check whether the file already exists
                key_ids = None
                try:
                    document = self.search_by_id(self.opensearch_client, doc_id)
                    if document:
                        if logflag:
                            logger.info(f"[ upload ] File {file.filename} already exists.")
                        key_ids = document["_id"]
                except Exception as e:
                    logger.info(f"[ upload ] File {file.filename} does not exist.")
                if key_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded file {file.filename} already exists. Please change file name.",
                    )

                save_path = self.upload_folder + encode_file
                await save_content_to_local_disk(save_path, file)
                await self.ingest_data_to_opensearch(
                    DocPath(
                        path=save_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        process_table=process_table,
                        table_strategy=table_strategy,
                    )
                )
                uploaded_files.append(save_path)
                if logflag:
                    logger.info(f"[ upload ] Successfully saved file {save_path}")

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
                doc_id = "file:" + encoded_link + ".txt"
                if logflag:
                    logger.info(f"[ upload ] processing link {doc_id}")

                # check whether the link file already exists
                key_ids = None
                try:
                    document = self.search_by_id(self.opensearch_client, doc_id)
                    if document:
                        if logflag:
                            logger.info(f"[ upload ] Link {link} already exists.")
                        key_ids = document["_id"]
                except Exception as e:
                    logger.info(f"[ upload ] Link {link} does not exist. Keep storing.")
                if key_ids:
                    raise HTTPException(
                        status_code=400, detail=f"Uploaded link {link} already exists. Please change another link."
                    )

                save_path = self.upload_folder + encoded_link + ".txt"
                content = parse_html([link])[0][0]
                await save_content_to_local_disk(save_path, content)
                await self.ingest_data_to_opensearch(
                    DocPath(
                        path=save_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        process_table=process_table,
                        table_strategy=table_strategy,
                    )
                )
            if logflag:
                logger.info(f"[ upload ] Successfully saved link list {link_list}")
            return {"status": 200, "message": "Data preparation succeeded"}

        raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

    async def get_files(self):
        """Get file structure from opensearch database in the format of
        {
            "name": "File Name",
            "id": "File Name",
            "type": "File",
            "parent": "",
        }"""
        if logflag:
            logger.info("[ get ] start to get file structure")

        offset = 0
        file_list = []

        # check index existence
        res = self.check_index_existence(self.opensearch_client, Config.KEY_INDEX_NAME)
        if not res:
            if logflag:
                logger.info(f"[ get ] index {Config.KEY_INDEX_NAME} does not exist")
            return file_list

        while True:
            response = self.search_all_documents(Config.KEY_INDEX_NAME, offset, Config.SEARCH_BATCH_SIZE)
            if response is None:
                break

            def format_opensearch_results(response, file_list):
                for document in response["documents"]:
                    file_id = document["_id"]
                    file_list.append({"name": file_id, "id": file_id, "type": "File", "parent": ""})

            format_opensearch_results(response, file_list)
            offset += Config.SEARCH_BATCH_SIZE
            # last batch
            if (len(response) - 1) // 2 < Config.SEARCH_BATCH_SIZE:
                break
        if logflag:
            logger.info(f"[get] final file_list: {file_list}")
        return file_list

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file according to `file_path`.

        `file_path`:
            - specific file path (e.g. /path/to/file.txt)
            - "all": delete all files uploaded
        """
        # delete all uploaded files
        if file_path == "all":
            if logflag:
                logger.info("[ delete ] delete all files")

            # drop index KEY_INDEX_NAME
            if self.check_index_existence(self.opensearch_client, Config.KEY_INDEX_NAME):
                try:
                    assert self.drop_index(client=self.opensearch_client, index_name=Config.KEY_INDEX_NAME)
                except Exception as e:
                    if logflag:
                        logger.info(f"[ delete ] {e}. Fail to drop index {Config.KEY_INDEX_NAME}.")
                    raise HTTPException(status_code=500, detail=f"Fail to drop index {Config.KEY_INDEX_NAME}.")
            else:
                logger.info(f"[ delete ] Index {Config.KEY_INDEX_NAME} does not exits.")

            # drop index INDEX_NAME
            if self.check_index_existence(self.opensearch_client, Config.INDEX_NAME):
                try:
                    assert self.drop_index(client=self.opensearch_client, index_name=Config.INDEX_NAME)
                except Exception as e:
                    if logflag:
                        logger.info(f"[ delete ] {e}. Fail to drop index {Config.INDEX_NAME}.")
                    raise HTTPException(status_code=500, detail=f"Fail to drop index {Config.INDEX_NAME}.")
            else:
                if logflag:
                    logger.info(f"[ delete ] Index {Config.INDEX_NAME} does not exits.")

            # delete files on local disk
            try:
                remove_folder_with_ignore(self.upload_folder)
            except Exception as e:
                if logflag:
                    logger.info(f"[ delete ] {e}. Fail to delete {self.upload_folder}.")
                raise HTTPException(status_code=500, detail=f"Fail to delete {self.upload_folder}.")

            if logflag:
                logger.info("[ delete ] successfully delete all files.")
            create_upload_folder(self.upload_folder)
            if logflag:
                logger.info({"status": True})
            return {"status": True}
        else:
            raise HTTPException(status_code=404, detail="Single file deletion is not implemented yet")
