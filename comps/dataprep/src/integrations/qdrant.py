# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import List, Optional, Union

from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter
from qdrant_client import QdrantClient

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DataprepRequest
from comps.dataprep.src.utils import (
    document_loader,
    encode_filename,
    get_separators,
    get_tables_result,
    parse_html_new,
    save_content_to_local_disk,
)

logger = CustomLogger("opea_dataprep_qdrant")
logflag = os.getenv("LOGFLAG", False)


# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag-qdrant")

# LLM/Embedding endpoints
TGI_LLM_ENDPOINT = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
TGI_LLM_ENDPOINT_NO_RAG = os.getenv("TGI_LLM_ENDPOINT_NO_RAG", "http://localhost:8081")
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
# Huggingface API token for TEI embedding endpoint
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


@OpeaComponentRegistry.register("OPEA_DATAPREP_QDRANT")
class OpeaQdrantDataprep(OpeaComponent):
    """Dataprep component for Qdrant ingestion and search services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.upload_folder = "./uploaded_files/"
        # Create vectorstore
        if TEI_EMBEDDING_ENDPOINT:
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
            # create embeddings using TEI endpoint service
            self.embedder = HuggingFaceInferenceAPIEmbeddings(
                api_key=HF_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        else:
            # create embeddings using local embedding model
            self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        # Perform health check
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaQdrantDataprep health check failed.")

    def check_health(self) -> bool:
        """Checks the health of the Qdrant service."""
        if self.embedder is None:
            logger.error("Qdrant embedder is not initialized.")
            return False

        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            logger.info(client.info())
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def invoke(self, *args, **kwargs):
        pass

    async def ingest_data_to_qdrant(self, doc_path: DocPath):
        """Ingest document to Qdrant."""
        path = doc_path.path
        if logflag:
            logger.info(f"Parsing document {path}.")

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
            logger.info("Done preprocessing. Created ", len(chunks), " chunks of the original file.")

        # Batch size
        batch_size = 32
        num_chunks = len(chunks)
        for i in range(0, num_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = batch_chunks

            _ = Qdrant.from_texts(
                texts=batch_texts,
                embedding=self.embedder,
                collection_name=COLLECTION_NAME,
                host=QDRANT_HOST,
                port=QDRANT_PORT,
            )
            if logflag:
                logger.info(f"Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")

        return True

    async def ingest_files(
        self,
        input: DataprepRequest,
    ):
        """Ingest files/links content into qdrant database.

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

        if files:
            if not isinstance(files, list):
                files = [files]
            uploaded_files = []
            for file in files:
                encode_file = encode_filename(file.filename)
                save_path = self.upload_folder + encode_file
                await save_content_to_local_disk(save_path, file)
                await self.ingest_data_to_qdrant(
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
                    logger.info(f"Successfully saved file {save_path}")
            result = {"status": 200, "message": "Data preparation succeeded"}
            if logflag:
                logger.info(result)
            return result

        if link_list:
            link_list = json.loads(link_list)  # Parse JSON string to list
            if not isinstance(link_list, list):
                raise HTTPException(status_code=400, detail="link_list should be a list.")
            for link in link_list:
                encoded_link = encode_filename(link)
                save_path = self.upload_folder + encoded_link + ".txt"
                content = parse_html_new([link], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                try:
                    await save_content_to_local_disk(save_path, content)
                    await self.ingest_data_to_qdrant(
                        DocPath(
                            path=save_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            process_table=process_table,
                            table_strategy=table_strategy,
                        )
                    )
                except json.JSONDecodeError:
                    raise HTTPException(status_code=500, detail="Fail to ingest data into qdrant.")

                if logflag:
                    logger.info(f"Successfully saved link {link}")

            result = {"status": 200, "message": "Data preparation succeeded"}
            if logflag:
                logger.info(result)
            return result

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
