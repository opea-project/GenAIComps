# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import psycopg2
from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import PGVector

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.dataprep.src.utils import (
    create_upload_folder,
    document_loader,
    encode_filename,
    get_file_structure,
    get_separators,
    parse_html_new,
    remove_folder_with_ignore,
    save_content_to_local_disk,
)

logger = CustomLogger("opea_dataprep_pgvector")
logflag = os.getenv("LOGFLAG", False)

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# TEI Embedding endpoints
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
# Huggingface API token for TEI embedding endpoint
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING", "localhost")

# Vector Index Configuration
INDEX_NAME = os.getenv("INDEX_NAME", "rag-pgvector")

# chunk parameters
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 1500)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 100)


@OpeaComponentRegistry.register("OPEA_DATAPREP_PGVECTOR")
class OpeaPgvectorDataprep(OpeaComponent):
    """Dataprep component for PgVector ingestion and search services."""

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
            self.embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

        # Perform health check
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaPgvectorDataprep health check failed.")

    def check_health(self) -> bool:
        """Checks the health of the PgVector service."""
        try:
            result = urlparse(PG_CONNECTION_STRING)
            username = result.username
            password = result.password
            database = result.path[1:]
            hostname = result.hostname
            port = result.port

            psycopg2.connect(database=database, user=username, password=password, host=hostname, port=port)
            return True
        except psycopg2.Error as e:
            if logflag:
                logger.info(f"Error connect to PG vectorstore: {e}")
            return False

        except Exception as e:
            if logflag:
                logger.info(f"An unexpected error occurred: {e}")
            return False

    def invoke(self, *args, **kwargs):
        pass

    async def save_file_to_local_disk(self, save_path: str, file):
        save_path = Path(save_path)
        with save_path.open("wb") as fout:
            try:
                content = await file.read()
                fout.write(content)
            except Exception as e:
                if logflag:
                    logger.info(f"Write file failed. Exception: {e}")
                raise HTTPException(status_code=500, detail=f"Write file {save_path} failed. Exception: {e}")

    def delete_embeddings(self, doc_name):
        """Get all ids from a vectorstore."""
        try:
            result = urlparse(PG_CONNECTION_STRING)
            username = result.username
            password = result.password
            database = result.path[1:]
            hostname = result.hostname
            port = result.port

            connection = psycopg2.connect(database=database, user=username, password=password, host=hostname, port=port)

            # Create a cursor object to execute SQL queries

            if logflag:
                logger.info(f"Deleting {doc_name} from vectorstore")

            cur = connection.cursor()
            if doc_name == "all":
                cur.execute(
                    "DELETE FROM langchain_pg_collection lpe WHERE lpe.name = %(index_name)s",
                    {"index_name": INDEX_NAME},
                )
            else:
                cur.execute(
                    "DELETE  FROM langchain_pg_embedding lpe WHERE lpe.uuid in (SELECT lpc.uuid\
                        FROM langchain_pg_embedding lpc where lpc.cmetadata ->> 'doc_name' = %(doc_name)s)",
                    {"doc_name": doc_name},
                )

            connection.commit()  # commit the transaction
            cur.close()

            return True

        except psycopg2.Error as e:
            if logflag:
                logger.info(f"Error deleting document from vectorstore: {e}")
            return False

        except Exception as e:
            if logflag:
                logger.info(f"An unexpected error occurred: {e}")
            return False

    async def ingest_doc_to_pgvector(self, doc_path: DocPath):
        """Ingest document to PGVector."""
        doc_path = doc_path.path
        if logflag:
            logger.info(f"Parsing document {doc_path}.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True, separators=get_separators()
        )

        content = await document_loader(doc_path)

        structured_types = [".xlsx", ".csv", ".json", "jsonl"]
        _, ext = os.path.splitext(doc_path)

        if ext in structured_types:
            chunks = content
        else:
            chunks = text_splitter.split_text(content)

        if logflag:
            logger.info("Done preprocessing. Created ", len(chunks), " chunks of the original file.")
            logger.info("PG Connection", PG_CONNECTION_STRING)
        metadata = [dict({"doc_name": str(doc_path)})]

        # Batch size
        batch_size = 32
        num_chunks = len(chunks)
        for i in range(0, num_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = batch_chunks

            _ = PGVector.from_texts(
                texts=batch_texts,
                embedding=self.embedder,
                metadatas=metadata,
                collection_name=INDEX_NAME,
                connection_string=PG_CONNECTION_STRING,
            )
            if logflag:
                logger.info(f"Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")
        return True

    async def ingest_link_to_pgvector(self, link_list: List[str]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True, separators=get_separators()
        )

        for link in link_list:
            texts = []
            content = parse_html_new([link], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            if logflag:
                logger.info(f"[ ingest link ] link: {link} content: {content}")
            encoded_link = encode_filename(link)
            save_path = self.upload_folder + encoded_link + ".txt"
            doc_path = self.upload_folder + link + ".txt"
            if logflag:
                logger.info(f"[ ingest link ] save_path: {save_path}")
            await save_content_to_local_disk(save_path, content)
            metadata = [dict({"doc_name": str(doc_path)})]

            chunks = text_splitter.split_text(content)

            batch_size = 32
            num_chunks = len(chunks)
            for i in range(0, num_chunks, batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_texts = batch_chunks

                _ = PGVector.from_texts(
                    texts=batch_texts,
                    embedding=self.embedder,
                    metadatas=metadata,
                    collection_name=INDEX_NAME,
                    connection_string=PG_CONNECTION_STRING,
                )
                if logflag:
                    logger.info(f"Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")

        return True

    async def ingest_files(
        self,
        files: Optional[Union[UploadFile, List[UploadFile]]] = File(None),
        link_list: Optional[str] = Form(None),
        chunk_size: int = Form(1500),
        chunk_overlap: int = Form(100),
        process_table: bool = Form(False),
        table_strategy: str = Form("fast"),
        ingest_from_graphDB: bool = Form(False),
    ):
        """Ingest files/links content into pgvector database.

        Save in the format of vector[768].
        Returns '{"status": 200, "message": "Data preparation succeeded"}' if successful.
        Args:
            files (Union[UploadFile, List[UploadFile]], optional): A file or a list of files to be ingested. Defaults to File(None).
            link_list (str, optional): A list of links to be ingested. Defaults to Form(None).
            chunk_size (int, optional): The size of the chunks to be split. Defaults to Form(1500).
            chunk_overlap (int, optional): The overlap between chunks. Defaults to Form(100).
            process_table (bool, optional): Whether to process tables in PDFs. Defaults to Form(False).
            table_strategy (str, optional): The strategy to process tables in PDFs. Defaults to Form("fast").
        """
        if logflag:
            logger.info(f"files:{files}")
            logger.info(f"link_list:{link_list}")
        if files and link_list:
            raise HTTPException(status_code=400, detail="Provide either a file or a string list, not both.")

        if files:
            if not isinstance(files, list):
                files = [files]

            if not os.path.exists(self.upload_folder):
                Path(self.upload_folder).mkdir(parents=True, exist_ok=True)
            for file in files:
                save_path = self.upload_folder + file.filename
                await self.save_file_to_local_disk(save_path, file)

                await self.ingest_doc_to_pgvector(DocPath(path=save_path))
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
                await self.ingest_link_to_pgvector(link_list)
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
        """Get file structure from pgvector database in the format of
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
        if file_path == "all":
            if logflag:
                logger.info("[dataprep - del] delete all files")
            remove_folder_with_ignore(self.upload_folder)
            assert self.delete_embeddings(file_path)
            if logflag:
                logger.info("[dataprep - del] successfully delete all files.")
            create_upload_folder(self.upload_folder)
            if logflag:
                logger.info({"status": True})
            return {"status": True}

        delete_path = Path(self.upload_folder + "/" + encode_filename(file_path))
        doc_path = self.upload_folder + file_path
        if logflag:
            logger.info(f"[dataprep - del] delete_path: {delete_path}")

        # partially delete files/folders
        if delete_path.exists():
            # delete file
            if delete_path.is_file():
                try:
                    assert self.delete_embeddings(doc_path)
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
