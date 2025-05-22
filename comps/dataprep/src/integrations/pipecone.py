# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import HTMLHeaderTextSplitter
from pinecone import Pinecone, ServerlessSpec

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DataprepRequest
from comps.dataprep.src.utils import (
    create_upload_folder,
    document_loader,
    encode_filename,
    get_file_structure,
    get_separators,
    get_tables_result,
    parse_html_new,
    remove_folder_with_ignore,
    save_content_to_local_disk,
)

logger = CustomLogger("opea_dataprep_pinecone")
logflag = os.getenv("LOGFLAG", False)

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-test")

# LLM/Embedding endpoints
TGI_LLM_ENDPOINT = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
TGI_LLM_ENDPOINT_NO_RAG = os.getenv("TGI_LLM_ENDPOINT_NO_RAG", "http://localhost:8081")
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
# Huggingface API token for TEI embedding endpoint
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


@OpeaComponentRegistry.register("OPEA_DATAPREP_PINECONE")
class OpeaPineConeDataprep(OpeaComponent):
    """Dataprep component for Pinecone ingestion and search services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.upload_folder = "./uploaded_files/"
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
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Perform health check
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaPineConeDataprep health check failed.")

    def check_health(self) -> bool:
        """Checks the health of the Pinecone service."""
        if self.pc is None:
            logger.error("Pinecone client is not initialized.")
            return False

        try:
            # Perform a simple health check via listing indexes
            self.pc.list_indexes()
            return True
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
            return False

    def invoke(self, *args, **kwargs):
        pass

    def check_index_existance(self):
        if logflag:
            logger.info(f"[ check index existence ] checking {PINECONE_INDEX_NAME}")

        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing_indexes:
            if logflag:
                logger.info("[ check index existence ] index does not exist")
            return None
        else:
            return True

    def create_index(self, client):
        if logflag:
            logger.info(f"[ create index ] creating index {PINECONE_INDEX_NAME}")
        try:
            client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            if logflag:
                logger.info(f"[ create index ] index {PINECONE_INDEX_NAME} successfully created")
        except Exception as e:
            if logflag:
                logger.info(f"[ create index ] fail to create index {PINECONE_INDEX_NAME}: {e}")
            return False
        return True

    def drop_index(self, index_name):
        if logflag:
            logger.info(f"[ drop index ] dropping index {index_name}")
        try:
            self.pc.delete_index(index_name)
            if logflag:
                logger.info(f"[ drop index ] index {index_name} deleted")
        except Exception as e:
            if logflag:
                logger.info(f"[ drop index ] index {index_name} delete failed: {e}")
            return False
        return True

    async def ingest_data_to_pinecone(self, doc_path: DocPath):
        """Ingest document to Pinecone."""
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
            logger.info(f"Done preprocessing. Created {len(chunks)} chunks of the original file.")

        # Checking Index existence
        if not self.check_index_existance():
            # Creating the index
            self.create_index(self.pc)
            if logflag:
                logger.info(f"Successfully created the index {PINECONE_INDEX_NAME}")

        # Batch size
        batch_size = 32
        num_chunks = len(chunks)
        file_ids = []

        for i in range(0, num_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = batch_chunks

            vectorstore = PineconeVectorStore.from_texts(
                texts=batch_texts,
                embedding=self.embedder,
                index_name=PINECONE_INDEX_NAME,
            )
            if logflag:
                logger.info(f"Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")

    async def ingest_link_to_pinecone(self, link_list: List[str], chunk_size, chunk_overlap):
        # Checking Index existence
        if not self.check_index_existance():
            # Creating the index
            self.create_index(self.pc)
            if logflag:
                logger.info(f"Successfully created the index {PINECONE_INDEX_NAME}")

        # save link contents and doc_ids one by one
        for link in link_list:
            content = parse_html_new([link], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if logflag:
                logger.info(f"[ ingest link ] link: {link} content: {content}")
            encoded_link = encode_filename(link)
            save_path = self.upload_folder + encoded_link + ".txt"
            if logflag:
                logger.info(f"[ ingest link ] save_path: {save_path}")
            await save_content_to_local_disk(save_path, content)

            vectorstore = PineconeVectorStore.from_texts(
                texts=content,
                embedding=self.embedder,
                index_name=PINECONE_INDEX_NAME,
            )

        return True

    async def ingest_files(self, input: DataprepRequest):
        """Ingest files/links content into pipecone database.

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
                await self.ingest_data_to_pinecone(
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
            try:
                link_list = json.loads(link_list)  # Parse JSON string to list
                if not isinstance(link_list, list):
                    raise HTTPException(status_code=400, detail="link_list should be a list.")
                await self.ingest_link_to_pinecone(link_list, chunk_size, chunk_overlap)
                result = {"status": 200, "message": "Data preparation succeeded"}
                if logflag:
                    logger.info(f"Successfully saved link list {link_list}")
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

        if not Path(self.upload_folder).exists():
            if logflag:
                logger.info("No file uploaded, return empty list.")
            return []

        file_content = get_file_structure(self.upload_folder)
        if logflag:
            logger.info(file_content)
        return file_content

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file according to `file_path`.

        `file_path`:
            - specific file path (e.g. /path/to/file.txt)
            - "all": delete all files uploaded
        """
        # delete all uploaded files
        if file_path == "all":
            if logflag:
                logger.info("[dataprep - del] delete all files")
            remove_folder_with_ignore(self.upload_folder)
            assert self.drop_index(index_name=PINECONE_INDEX_NAME)
            if logflag:
                logger.info("[dataprep - del] successfully delete all files.")
            create_upload_folder(self.upload_folder)
            if logflag:
                logger.info('{"status": True}')
            return {"status": True}
        else:
            raise HTTPException(status_code=404, detail="Single file deletion is not implemented yet")
