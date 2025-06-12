# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from elasticsearch import Elasticsearch
from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DataprepRequest
from comps.dataprep.src.utils import (
    create_upload_folder,
    document_loader,
    encode_filename,
    get_file_structure,
    get_separators,
    get_tables_result,
    parse_html,
    remove_folder_with_ignore,
    save_content_to_local_disk,
)

logger = CustomLogger("opea_dataprep_elasticsearch")
logflag = os.getenv("LOGFLAG", False)

ES_CONNECTION_STRING = os.getenv("ES_CONNECTION_STRING", "http://localhost:9200")
UPLOADED_FILES_PATH = os.getenv("UPLOADED_FILES_PATH", "./uploaded_files/")

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# TEI Embedding endpoints
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
# Huggingface API token for TEI embedding endpoint
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Vector Index Configuration
INDEX_NAME = os.getenv("INDEX_NAME", "rag-elastic")

# chunk parameters
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 1500)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 100)


@OpeaComponentRegistry.register("OPEA_DATAPREP_ELASTICSEARCH")
class OpeaElasticSearchDataprep(OpeaComponent):
    """Dataprep component for ElasticSearch ingestion and search services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.es_client = Elasticsearch(hosts=ES_CONNECTION_STRING)
        self.es_store = self.get_elastic_store(self.get_embedder())
        self.create_index()

        # Perform health check
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaElasticSearchDataprep health check failed.")

    def check_health(self) -> bool:
        """Checks the health of the ElasticSearch service."""
        if self.es_client is None:
            logger.error("ElasticSearch client is not initialized.")
            return False

        return True

    def invoke(self, *args, **kwargs):
        pass

    def create_index(self) -> None:
        if not self.es_client.indices.exists(index=INDEX_NAME):
            self.es_client.indices.create(index=INDEX_NAME)

    def get_embedder(self) -> Union[HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings]:
        """Obtain required Embedder."""
        if TEI_EMBEDDING_ENDPOINT:
            if not HUGGINGFACEHUB_API_TOKEN:
                raise HTTPException(
                    status_code=400,
                    detail="You MUST offer the `HUGGINGFACEHUB_API_TOKEN` and the `EMBED_MODEL` when using `TEI_EMBEDDING_ENDPOINT`.",
                )
            import requests

            response = requests.get(TEI_EMBEDDING_ENDPOINT + "/info")
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available."
                )
            model_id = response.json()["model_id"]
            embedder = HuggingFaceInferenceAPIEmbeddings(
                api_key=HUGGINGFACEHUB_API_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
            return embedder
        else:
            return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    def get_elastic_store(
        self, embedder: Union[HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings]
    ) -> ElasticsearchStore:
        """Get Elasticsearch vector store."""
        return ElasticsearchStore(index_name=INDEX_NAME, embedding=embedder, es_connection=self.es_client)

    def delete_embeddings(self, doc_name: str) -> bool:
        """Delete documents from Elasticsearch."""
        try:
            if doc_name == "all":
                if logflag:
                    logger.info("Deleting all documents from vectorstore")

                query = {"query": {"match_all": {}}}
            else:
                if logflag:
                    logger.info(f"Deleting {doc_name} from vectorstore")

                query = {"query": {"match": {"metadata.doc_name": {"query": doc_name, "operator": "AND"}}}}

            self.es_client.delete_by_query(index=INDEX_NAME, body=query)
            return True

        except Exception as e:
            if logflag:
                logger.info(f"An unexpected error occurred: {e}")

            return False

    def search_by_filename(self, file_name: str) -> bool:
        """Search Elasticsearch by file name."""

        query = {"query": {"match": {"metadata.doc_name": {"query": file_name, "operator": "AND"}}}}
        results = self.es_client.search(index=INDEX_NAME, body=query)

        if logflag:
            logger.info(f"[ search by file ] searched by {file_name}")
            logger.info(f"[ search by file ] {len(results['hits'])} results: {results}")

        return results["hits"]["total"]["value"] > 0

    async def ingest_doc_to_elastic(self, doc_path: DocPath) -> None:
        """Ingest documents to Elasticsearch."""

        path = doc_path.path
        file_name = path.split("/")[-1]
        if logflag:
            logger.info(f"Parsing document {path}, file name: {file_name}.")

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

        structured_types = [".xlsx", ".csv", ".json", "jsonl"]
        _, ext = os.path.splitext(path)

        if ext in structured_types:
            chunks = content
        else:
            chunks = text_splitter.split_text(content)

        if doc_path.process_table and path.endswith(".pdf"):
            table_chunks = get_tables_result(path, doc_path.table_strategy)
            logger.info(f"table chunks: {table_chunks}")
            if table_chunks:
                chunks = chunks + table_chunks
            else:
                logger.info(f"No table chunks found in {path}.")

        if logflag:
            logger.info(f"Done preprocessing. Created {len(chunks)} chunks of the original file.")

        batch_size = 32
        num_chunks = len(chunks)

        metadata = dict({"doc_name": str(file_name)})

        for i in range(0, num_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = batch_chunks

            documents = [Document(page_content=text, metadata=metadata) for text in batch_texts]
            _ = self.es_store.add_documents(documents=documents)
            if logflag:
                logger.info(f"Processed batch {i // batch_size + 1}/{(num_chunks - 1) // batch_size + 1}")

    async def ingest_link_to_elastic(self, link_list: List[str]) -> None:
        """Ingest data scraped from website links into Elasticsearch."""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
            separators=get_separators(),
        )

        batch_size = 32

        for link in link_list:
            content = parse_html([link])[0][0]
            if logflag:
                logger.info(f"[ ingest link ] link: {link} content: {content}")

            encoded_link = encode_filename(link)
            save_path = UPLOADED_FILES_PATH + encoded_link + ".txt"
            doc_path = UPLOADED_FILES_PATH + link + ".txt"
            if logflag:
                logger.info(f"[ ingest link ] save_path: {save_path}")

            await save_content_to_local_disk(save_path, content)

            chunks = text_splitter.split_text(content)

            num_chunks = len(chunks)
            metadata = [dict({"doc_name": str(doc_path)})]

            for i in range(0, num_chunks, batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_texts = batch_chunks

                documents = [Document(page_content=text, metadata=metadata) for text in batch_texts]
                _ = self.es_store.add_documents(documents=documents)

                if logflag:
                    logger.info(f"Processed batch {i // batch_size + 1}/{(num_chunks - 1) // batch_size + 1}")

    async def ingest_files(self, input: DataprepRequest):
        """Ingest files/links content into ElasticSearch database.

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
            logger.info(f"files:{files}")
            logger.info(f"link_list:{link_list}")

        if files and link_list:
            raise HTTPException(status_code=400, detail="Provide either a file or a string list, not both.")

        if files:
            if not isinstance(files, list):
                files = [files]

            if not os.path.exists(UPLOADED_FILES_PATH):
                Path(UPLOADED_FILES_PATH).mkdir(parents=True, exist_ok=True)

            for file in files:
                encode_file = encode_filename(file.filename)
                save_path = UPLOADED_FILES_PATH + encode_file
                filename = save_path.split("/")[-1]

                try:
                    exists = self.search_by_filename(filename)
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed when searching in Elasticsearch for file {file.filename}.",
                    )

                if exists:
                    if logflag:
                        logger.info(f"[ upload ] File {file.filename} already exists.")

                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded file {file.filename} already exists. Please change file name.",
                    )

                await save_content_to_local_disk(save_path, file)

                await self.ingest_doc_to_elastic(
                    DocPath(
                        path=save_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        process_table=process_table,
                        table_strategy=table_strategy,
                    )
                )
                if logflag:
                    logger.info(f"Successfully saved file {save_path}")

            result = {"status": 200, "message": "Data preparation succeeded"}

            if logflag:
                logger.info(result)
            return result

        if link_list:
            try:
                link_list = json.loads(link_list)  # Parse JSON string to list
                if not isinstance(link_list, list):
                    raise HTTPException(status_code=400, detail="link_list should be a list.")

                await self.ingest_link_to_elastic(link_list)

                if logflag:
                    logger.info(f"Successfully saved link list {link_list}")

                result = {"status": 200, "message": "Data preparation succeeded"}

                if logflag:
                    logger.info(result)
                return result

            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format for link_list.")

        raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

    async def get_files(self):
        """Get file structure from pipecone database in the format of
        {
            "name": "File Name",
            "id": "File Name",
            "type": "File",
            "parent": "",
        }"""
        if logflag:
            logger.info("[ dataprep - get file ] start to get file structure")

        if not Path(UPLOADED_FILES_PATH).exists():
            if logflag:
                logger.info("No file uploaded, return empty list.")
            return []

        file_content = get_file_structure(UPLOADED_FILES_PATH)

        if logflag:
            logger.info(file_content)

        return file_content

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file according to `file_path`.

        `file_path`:
            - specific file path (e.g. /path/to/file.txt)
            - "all": delete all files uploaded
        """
        if file_path == "all":
            if logflag:
                logger.info("[dataprep - del] delete all files")
            remove_folder_with_ignore(UPLOADED_FILES_PATH)
            assert self.delete_embeddings(file_path)
            if logflag:
                logger.info("[dataprep - del] successfully delete all files.")
            create_upload_folder(UPLOADED_FILES_PATH)
            if logflag:
                logger.info({"status": True})
            return {"status": True}

        delete_path = Path(UPLOADED_FILES_PATH + "/" + encode_filename(file_path))

        if logflag:
            logger.info(f"[dataprep - del] delete_path: {delete_path}")

        if delete_path.exists():
            # delete file
            if delete_path.is_file():
                try:
                    assert self.delete_embeddings(file_path)
                    delete_path.unlink()
                except Exception as e:
                    if logflag:
                        logger.info(f"[dataprep - del] fail to delete file {delete_path}: {e}")
                        logger.info({"status": False})
                    return {"status": False}
            # delete folder
            else:
                if logflag:
                    logger.info("[dataprep - del] delete folder is not supported for now.")
                    logger.info({"status": False})
                return {"status": False}
            if logflag:
                logger.info({"status": True})
            return {"status": True}
        else:
            raise HTTPException(status_code=404, detail="File/folder not found. Please check del_path.")
