# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import List, Optional, Union

from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_vdms.vectorstores import VDMS, VDMS_Client

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DataprepRequest
from comps.dataprep.src.utils import (
    create_upload_folder,
    document_loader,
    encode_filename,
    get_separators,
    get_tables_result,
    parse_html_new,
    save_content_to_local_disk,
)

logger = CustomLogger("opea_dataprep_vdms")
logflag = os.getenv("LOGFLAG", False)


def getEnv(key, default_value=None):
    env_value = os.getenv(key, default=default_value)
    return env_value


# Embedding model
EMBED_MODEL = getEnv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# VDMS configuration
VDMS_HOST = getEnv("VDMS_HOST", "localhost")
VDMS_PORT = int(getEnv("VDMS_PORT", 55555))
COLLECTION_NAME = getEnv("COLLECTION_NAME", "rag-vdms")
SEARCH_ENGINE = getEnv("SEARCH_ENGINE", "FaissFlat")
DISTANCE_STRATEGY = getEnv("DISTANCE_STRATEGY", "L2")

# LLM/Embedding endpoints
TGI_LLM_ENDPOINT = getEnv("TGI_LLM_ENDPOINT", "http://localhost:8080")
TGI_LLM_ENDPOINT_NO_RAG = getEnv("TGI_LLM_ENDPOINT_NO_RAG", "http://localhost:8081")
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
# Huggingface API token for TEI embedding endpoint
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# chunk parameters
CHUNK_SIZE = getEnv("CHUNK_SIZE", 1500)
CHUNK_OVERLAP = getEnv("CHUNK_OVERLAP", 100)


@OpeaComponentRegistry.register("OPEA_DATAPREP_VDMS")
class OpeaVdmsDataprep(OpeaComponent):
    """Dataprep component for VDMS ingestion and search services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.upload_folder = "./uploaded_files/"
        create_upload_folder(self.upload_folder)
        self.client = VDMS_Client(VDMS_HOST, int(VDMS_PORT))
        # Create vectorstore
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
            # create embeddings using TEI endpoint service
            self.embedder = HuggingFaceInferenceAPIEmbeddings(
                api_key=HUGGINGFACEHUB_API_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        else:
            # create embeddings using local embedding model
            self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        # Perform health check
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaVdmsDataprep health check failed.")

    def check_health(self) -> bool:
        """Checks the health of the VDMS service."""
        if self.client is None:
            logger.error("VDMS client is not initialized.")
            return False

        return True

    def invoke(self, *args, **kwargs):
        pass

    async def ingest_data_to_vdms(self, doc_path: DocPath):
        """Ingest document to VDMS."""
        path = doc_path.path
        print(f"Parsing document {doc_path}.")

        if path.endswith(".html"):
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ]
            text_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=doc_path.chunk_size, chunk_overlap=100, add_start_index=True, separators=get_separators()
            )

        content = await document_loader(path)
        chunks = text_splitter.split_text(content)
        if doc_path.process_table and path.endswith(".pdf"):
            table_chunks = get_tables_result(path, doc_path.table_strategy)
            logger.info(f"table chunks: {table_chunks}")
            if table_chunks:
                chunks = chunks + table_chunks
            else:
                logger.info(f"No table chunks found in {path}.")

        logger.info(f"Done preprocessing. Created {len(chunks)} chunks of the original pdf")

        # Batch size
        batch_size = 32
        num_chunks = len(chunks)
        for i in range(0, num_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = batch_chunks

            _ = VDMS.from_texts(
                client=self.client,
                embedding=self.embedder,
                collection_name=COLLECTION_NAME,
                distance_strategy=DISTANCE_STRATEGY,
                engine=SEARCH_ENGINE,
                texts=batch_texts,
            )
            logger.info(f"Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")

    async def ingest_files(
        self,
        input: DataprepRequest,
    ):
        """Ingest files/links content into VDMS database.

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

                save_path = self.upload_folder + encode_file
                await save_content_to_local_disk(save_path, file)
                await self.ingest_data_to_vdms(
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

                save_path = self.upload_folder + encoded_link + ".txt"
                content = parse_html_new([link], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                await save_content_to_local_disk(save_path, content)
                await self.ingest_data_to_vdms(
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
        """Get file structure from pipecone database in the format of
        {
            "name": "File Name",
            "id": "File Name",
            "type": "File",
            "parent": "",
        }"""
        pass

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file according to `file_path`.

        `file_path`:
            - specific file path (e.g. /path/to/file.txt)
            - "all": delete all files uploaded
        """
        pass
