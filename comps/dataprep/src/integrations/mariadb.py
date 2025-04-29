# Copyright (C) 2025 MariaDB Foundation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path
from typing import (
    List,
    Optional,
    Union,
)
from urllib.parse import urlparse

import mariadb
from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mariadb.vectorstores import MariaDBStore, MariaDBStoreSettings

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DataprepRequest
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


# A no-op logger that does nothing
class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


logger = CustomLogger("opea_dataprep_mariadbvector")
logflag = os.getenv("LOGFLAG", False)
if not logflag:
    logger = NullLogger()

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# TEI Embedding endpoints
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
# Huggingface API token for TEI embedding endpoint
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

MARIADB_CONNECTION_URL = os.getenv("MARIADB_CONNECTION_URL", "localhost")

# Vector Index Configuration
MARIADB_COLLECTION_NAME = os.getenv("MARIADB_COLLECTION_NAME", "rag_mariadbvector")

# chunk parameters
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 1500)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 100)


class DocumentsTable:
    """Table for storing documents."""

    def __init__(self, conn_args):
        self._table_name = "langchain_documents"
        self.conn_args = conn_args
        self.__post__init__()

    def __post__init__(self):
        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        """Create the documents table if it does not exist."""
        connection = mariadb.connect(**self.conn_args)
        cursor = connection.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id VARCHAR(32) PRIMARY KEY,
                name TEXT,
                embedding_ids JSON
            )
            """
        )
        connection.commit()
        cursor.close()
        connection.close()

    def insert_document_ids(self, id: str, name: str, embedding_ids: list):
        """Insert a document into the documents table."""
        connection = mariadb.connect(**self.conn_args)
        cursor = connection.cursor()
        cursor.execute(
            f"INSERT INTO {self._table_name} (id, name, embedding_ids) VALUES (?, ?, ?)",
            (id, name, json.dumps(embedding_ids)),
        )
        connection.commit()
        cursor.close()
        connection.close()

    def delete_document(self, id: str):
        """Delete a document from the documents table."""
        connection = mariadb.connect(**self.conn_args)
        cursor = connection.cursor()
        cursor.execute(f"DELETE FROM {self._table_name} WHERE id = ?", (id,))
        connection.commit()
        cursor.close()
        connection.close()

    def delete_all_documents(self):
        """Delete all documents from the documents table."""
        connection = mariadb.connect(**self.conn_args)
        cursor = connection.cursor()
        cursor.execute(f"DELETE FROM {self._table_name}")
        connection.commit()
        cursor.close()
        connection.close()

    def get_document_emb_ids(self, id: str):
        """Get the embedding ids for a document."""
        connection = mariadb.connect(**self.conn_args)
        cursor = connection.cursor()
        cursor.execute(f"SELECT embedding_ids FROM {self._table_name} WHERE id = ?", (id,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        if result:
            return json.loads(result[0])
        return None


@OpeaComponentRegistry.register("OPEA_DATAPREP_MARIADBVECTOR")
class OpeaMariaDBDataprep(OpeaComponent):
    """Dataprep component for MariaDBStore ingestion and search services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        args = urlparse(MARIADB_CONNECTION_URL)

        self.conn_args = {
            "host": args.hostname,
            "port": args.port,
            "user": args.username,
            "password": args.password,
            "database": args.path[1:],
        }

        self.upload_folder = Path("./uploaded_files/")
        self.embedder = self._initialize_embedder()

        # Perform health check
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaMariaDBDataprep health check failed.")

        self.store = self._initialize_client()
        self.documents = DocumentsTable(self.conn_args)

    def _initialize_embedder(self):
        if TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            logger.info(f"[ init embedder ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
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
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=HUGGINGFACEHUB_API_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        else:
            # create embeddings using local embedding model
            logger.info(f"[ init embedder ] LOCAL_EMBEDDING_MODEL:{EMBED_MODEL}")
            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return embeddings

    def _initialize_client(self) -> MariaDBStore:
        store = MariaDBStore(
            embeddings=self.embedder,
            collection_name=MARIADB_COLLECTION_NAME,
            datasource=MARIADB_CONNECTION_URL,
            config=MariaDBStoreSettings(lazy_init=True),
        )
        return store

    def check_health(self) -> bool:
        """Checks mariadb server health."""
        try:
            connection = mariadb.connect(**self.conn_args)
            return True
        except mariadb.Error as e:
            logger.error(f"Error connect to MariaDB Server: {e}")
            return False

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return False
        finally:
            try:
                connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

    def invoke(self, *args, **kwargs):
        pass

    async def _save_file_to_local_disk(self, save_path: Path, file):
        with save_path.open("wb") as fout:
            try:
                content = await file.read()
                fout.write(content)
            except Exception as e:
                logger.error(f"Write file failed. Exception: {e}")
                raise HTTPException(status_code=500, detail=f"Write file {save_path} failed. Exception: {e}")

    def _store_texts(self, doc_path: str, chunks: list[str], batch_size: int = 32):
        num_chunks = len(chunks)
        metadata = [{"doc_name": doc_path}]
        doc_id = hashlib.md5(str(doc_path).encode("utf-8")).hexdigest()
        doc_emb_ids = []
        for i in range(0, num_chunks, batch_size):
            batch_texts = chunks[i : i + batch_size]
            batch_ids = self.store.add_texts(
                texts=batch_texts,
                metadatas=metadata * len(batch_texts),
            )
            doc_emb_ids.extend(batch_ids)
        self.documents.insert_document_ids(id=doc_id, name=doc_path, embedding_ids=doc_emb_ids)
        if logflag:
            logger.info(f"Processed batch {i // batch_size + 1} / {(num_chunks - 1) // batch_size + 1}")

    async def _ingest_doc_to_mariadb(self, path: str):
        """Ingest document to mariadb."""
        doc_path = DocPath(path=path).path
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

        logger.info(f"Done preprocessing. Created {len(chunks)} chunks of the original file.")

        self._store_texts(doc_path, chunks)
        return True

    async def _ingest_link_to_mariadb(self, link_list: List[str]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True, separators=get_separators()
        )

        for link in link_list:
            content = parse_html_new([link], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            logger.info(f"[ ingest link ] link: {link} content: {content}")
            encoded_link = encode_filename(link)
            save_path = self.upload_folder / (encoded_link + ".txt")
            doc_path = self.upload_folder / (link + ".txt")
            logger.info(f"[ ingest link ] save_path: {save_path}")
            await save_content_to_local_disk(str(save_path), content)

            chunks = text_splitter.split_text(content)
            self._store_texts(str(doc_path), chunks)
        return True

    async def ingest_files(
        self,
        input: DataprepRequest,
    ):
        """Ingest files/links content into database.

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

        logger.info(f"files:{files}")
        logger.info(f"link_list:{link_list}")
        if files and link_list:
            raise HTTPException(status_code=400, detail="Provide either a file or a string list, not both.")

        if not files and not link_list:
            raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

        if files:
            if not isinstance(files, list):
                files = [files]

            self.upload_folder.mkdir(parents=True, exist_ok=True)
            for file in files:
                save_path = self.upload_folder / file.filename
                await self._save_file_to_local_disk(save_path, file)
                await self._ingest_doc_to_mariadb(str(save_path))
                logger.info(f"Successfully saved file {save_path}")

        if link_list:
            try:
                link_list = json.loads(link_list)  # Parse JSON string to list
                if not isinstance(link_list, list):
                    raise HTTPException(status_code=400, detail="link_list should be a list.")
                await self._ingest_link_to_mariadb(link_list)
                logger.info(f"Successfully saved link list {link_list}")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format for link_list.")

        result = {"status": 200, "message": "Data preparation succeeded"}
        logger.info(result)
        return result

    async def get_files(self):
        """Get file structure from database in the format of
        {
            "name": "File Name",
            "id": "File Name",
            "type": "File",
            "parent": "",
        }"""
        logger.info("[ dataprep - get file ] start to get file structure")

        if not self.upload_folder.exists():
            logger.info("No file uploaded, return empty list.")
            return []

        file_content = get_file_structure(str(self.upload_folder))
        logger.info(file_content)
        return file_content

    def _delete_embedding(self, doc_path: Path):
        doc_id = hashlib.md5(str(doc_path).encode("utf-8")).hexdigest()
        doc_emb_ids = self.documents.get_document_emb_ids(doc_id)
        self.store.delete(ids=doc_emb_ids)
        self.documents.delete_document(doc_id)

    def _delete_all_embeddings(self):
        self.store.delete_collection()
        self.documents.delete_all_documents()

    def _delete_all_files(self):
        """Delete all files in the upload folder."""
        logger.info("[dataprep - del] delete all files")
        remove_folder_with_ignore(str(self.upload_folder))
        self._delete_all_embeddings()
        logger.info("[dataprep - del] successfully delete all files.")
        create_upload_folder(str(self.upload_folder))

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file according to `file_path`.

        `file_path`:
            - specific file path (e.g. /path/to/file.txt)
            - "all": delete all files uploaded
        """
        if file_path == "all":
            self._delete_all_files()
            logger.info({"status": True})
            return {"status": True}

        # Case when file_path != all
        delete_path = self.upload_folder / encode_filename(file_path)
        logger.info(f"[dataprep - del] delete_path: {delete_path}")

        if not delete_path.exists():
            raise HTTPException(status_code=404, detail="File/folder not found. Please check del_path.")

        if not delete_path.is_file():
            logger.info("[dataprep - del] delete folder is not supported for now.")
            logger.info({"status": False})
            return {"status": False}
        self._delete_embedding(delete_path)
        delete_path.unlink()
        logger.info({"status": True})
        return {"status": True}
