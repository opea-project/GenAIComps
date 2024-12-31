# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings

from comps import CustomLogger, DocPath, OpeaComponent, ServiceType
from comps.dataprep.src.utils import (
    create_upload_folder,
    encode_filename,
    get_file_structure,
    load_file_to_chunks,
    parse_html_new,
    remove_folder_with_ignore,
    save_content_to_local_disk,
)
from comps.vectorstores.src.integrations.pinecone import OpeaPineconeVectorstores
from comps.vectorstores.src.opea_vectorstores_controller import OpeaVectorstoresController

from .config import EMBED_MODEL, TEI_EMBEDDING_ENDPOINT

logger = CustomLogger("prepare_doc_pinecone")
logflag = os.getenv("LOGFLAG", False)
upload_folder = "./uploaded_files/"


class OpeaPineconeDataprep(OpeaComponent):
    """A specialized dataprep component derived from OpeaComponent for Pinecone dataprep services.

    Attributes:
        client (Milvus): An instance of the Pinecone client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.embedder = self._initialize_embedder()
        self.db_controller = self._initialize_db_controller()

    def _initialize_embedder(self):
        if logflag:
            logger.info("[ initialize embedder ] initializing Pinecone embedder...")
        # Define embeddings according to server type (TEI, or local)
        if TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            if logflag:
                logger.info(f"[ Pinecone embedding ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
            embeddings = HuggingFaceHubEmbeddings(model=TEI_EMBEDDING_ENDPOINT)
        else:
            # create embeddings using local embedding model
            if logflag:
                logger.info(f"[ Pinecone embedding ] EMBED_MODEL:{EMBED_MODEL}")
            embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
        return embeddings

    def _initialize_db_controller(self) -> OpeaVectorstoresController:
        controller = OpeaVectorstoresController()
        pinecone_db = OpeaPineconeVectorstores(
            embedder=self.embedder, name="OpeaPineconeVectorstore", description="OPEA Pinecone Vectorstore Service"
        )
        controller.register(pinecone_db)
        controller.discover_and_activate()
        return controller

    def check_health(self) -> bool:
        """Checks the health of the dataprep service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ health check ] start to check health of pinecone")
        try:
            if self.db_controller.active_component.check_health():
                if logflag:
                    logger.info("[ health check ] Successfully connected to Pinecone!")
                return True
        except Exception as e:
            logger.info(f"[ health check ] Failed to connect to Milvus: {e}")
            return False

    def invoke(self, *args, **kwargs):
        pass

    async def ingest_files(
        self,
        files: Optional[Union[UploadFile, List[UploadFile]]] = File(None),
        link_list: Optional[str] = Form(None),
        chunk_size: int = Form(1500),
        chunk_overlap: int = Form(100),
        process_table: bool = Form(False),
        table_strategy: str = Form("fast"),
    ):
        """Ingest files/links content into pinecone database.

        Save in the format of vector[], the vector length depends on the emedding model type.
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
            logger.info(f"[ pinecone ingest ] files:{files}")
            logger.info(f"[ pinecone ingest ] link_list:{link_list}")

        if files:
            if not isinstance(files, list):
                files = [files]

            for file in files:
                if logflag:
                    logger.info(f"[ pinecone ingest ] processing file {file.filename}")

                # save file to local path
                save_path = upload_folder + encode_filename(file.filename)
                await save_content_to_local_disk(save_path, file)

                # load file contents to chunks
                chunks = load_file_to_chunks(
                    DocPath(
                        path=save_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        process_table=process_table,
                        table_strategy=table_strategy,
                    )
                )

                # ingest chunks into db
                res = await self.db_controller.ingest_chunks(file_name=encode_filename(file.filename), chunks=chunks)
                if not res:
                    if logflag:
                        logger.info(f"[ pinecone ingest ] Fail to ingest file {file.filename}")
                    raise HTTPException(status_code=400, detail=f"Fail to ingest {file.filename}. File already exists.")

                if logflag:
                    logger.info(f"[ pinecone ingest ] Successfully saved file {save_path}")

            result = {"status": 200, "message": "Data preparation succeeded"}
            return result

        if link_list:
            link_list = json.loads(link_list)  # Parse JSON string to list
            if not isinstance(link_list, list):
                raise HTTPException(status_code=400, detail=f"Link_list {link_list} should be a list.")
            for link in link_list:
                save_path = upload_folder + encode_filename(link) + ".txt"
                content = parse_html_new([link], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                await save_content_to_local_disk(save_path, content)

                chunks = load_file_to_chunks(
                    DocPath(
                        path=save_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        process_table=process_table,
                        table_strategy=table_strategy,
                    )
                )
                res = await self.db_controller.ingest_chunks(file_name=encode_filename(link) + ".txt", chunks=chunks)
                if not res:
                    if logflag:
                        logger.info(f"[ pinecone ingest ] Fail to ingest link {link}")
                    raise HTTPException(status_code=400, detail=f"Fail to ingest {link}. Link already exists.")

            if logflag:
                logger.info(f"[ pinecone ingest ] Successfully saved link list {link_list}")
            return {"status": 200, "message": "Data preparation succeeded"}

        raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

    async def get_files(self):
        """Get file structure from pinecone database in the format of
        {
            "name": "File Name",
            "id": "File Name",
            "type": "File",
            "parent": "",
        }"""

        if logflag:
            logger.info("[ pinecone get ] start to get file structure")

        if not Path(upload_folder).exists():
            if logflag:
                logger.info("No file uploaded, return empty list.")
            return []

        file_content = get_file_structure(upload_folder)
        if logflag:
            logger.info(file_content)
        return file_content

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file according to `file_path`.

        `file_path`:
            - specific file path (e.g. /path/to/file.txt)
            - "all": delete all files uploaded
        """
        if logflag:
            logger.info(f"[ pinecone delete ] delete files: {file_path}")

        # delete all uploaded files
        if file_path == "all":
            if logflag:
                logger.info("[ pinecone delete ] delete all files")

            # delete files on local disk
            try:
                remove_folder_with_ignore(upload_folder)
            except Exception as e:
                if logflag:
                    logger.info(f"[ pinecone delete ] {e}. Fail to delete {upload_folder}.")
                raise HTTPException(status_code=500, detail=f"Fail to delete {upload_folder}.")

            assert self.db_controller.delete_all_files()
            if logflag:
                logger.info("[ pinecone delete ] successfully delete all files.")

            create_upload_folder(upload_folder)
            if logflag:
                logger.info("[ pinecone delete ] new upload folder created.")
            return {"status": True}

        else:
            raise HTTPException(status_code=404, detail="Single file deletion is not implemented yet")
