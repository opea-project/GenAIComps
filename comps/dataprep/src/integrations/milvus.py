# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import json
import os
from pathlib import Path
from typing import List, Optional, Union

from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_milvus.vectorstores import Milvus
from langchain_text_splitters import HTMLHeaderTextSplitter

from comps import CustomLogger, DocPath, OpeaComponent, ServiceType
from comps.dataprep.src.utils import (
    create_upload_folder,
    document_loader,
    encode_filename,
    get_separators,
    get_tables_result,
    load_file_to_chunks,
    parse_html_new,
    remove_folder_with_ignore,
    save_content_to_local_disk,
)
from comps.vectorstores.src.integrations.milvus import OpeaMilvusVectorstores
from comps.vectorstores.src.opea_vectorstores_controller import OpeaVectorstoresController

from .config import (
    COLLECTION_NAME,
    LOCAL_EMBEDDING_MODEL,
    MILVUS_URI,
    MOSEC_EMBEDDING_ENDPOINT,
    MOSEC_EMBEDDING_MODEL,
    TEI_EMBEDDING_ENDPOINT,
)

logger = CustomLogger("milvus_dataprep")
logflag = os.getenv("LOGFLAG", False)
partition_field_name = "filename"
upload_folder = "./uploaded_files/"


class MosecEmbeddings(OpenAIEmbeddings):
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        batched_embeddings: List[List[float]] = []
        response = self.client.create(input=texts, **self._invocation_params)
        if not isinstance(response, dict):
            response = response.model_dump()
        batched_embeddings.extend(r["embedding"] for r in response["data"])

        _cached_empty_embedding: Optional[List[float]] = None

        def empty_embedding() -> List[float]:
            nonlocal _cached_empty_embedding
            if _cached_empty_embedding is None:
                average_embedded = self.client.create(input="", **self._invocation_params)
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                _cached_empty_embedding = average_embedded["data"][0]["embedding"]
            return _cached_empty_embedding

        return [e if e is not None else empty_embedding() for e in batched_embeddings]


class OpeaMilvusDataprep(OpeaComponent):
    """A specialized dataprep component derived from OpeaComponent for milvus dataprep services.

    Attributes:
        client (Milvus): An instance of the milvus client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.embedder = self._initialize_embedder()
        self.db_controller = self._initialize_db_controller()

    def _initialize_embedder(self):
        if logflag:
            logger.info("[ initialize embedder ] initializing milvus embedder...")
        # Define embeddings according to server type (TEI, MOSEC, or local)
        if MOSEC_EMBEDDING_ENDPOINT:
            # create embeddings using MOSEC endpoint service
            if logflag:
                logger.info(
                    f"[ milvus embedding ] MOSEC_EMBEDDING_ENDPOINT:{MOSEC_EMBEDDING_ENDPOINT}, MOSEC_EMBEDDING_MODEL:{MOSEC_EMBEDDING_MODEL}"
                )
            embeddings = MosecEmbeddings(model=MOSEC_EMBEDDING_MODEL)
        elif TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            if logflag:
                logger.info(f"[ milvus embedding ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
            embeddings = HuggingFaceHubEmbeddings(model=TEI_EMBEDDING_ENDPOINT)
        else:
            # create embeddings using local embedding model
            if logflag:
                logger.info(f"[ milvus embedding ] LOCAL_EMBEDDING_MODEL:{LOCAL_EMBEDDING_MODEL}")
            embeddings = HuggingFaceBgeEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
        return embeddings

    def _initialize_db_controller(self) -> OpeaVectorstoresController:
        controller = OpeaVectorstoresController()
        milvus_db = OpeaMilvusVectorstores(
            embedder=self.embedder, name="OpeaMilvusVectorstore", description="OPEA Milvus Vectorstore Service"
        )
        controller.register(milvus_db)
        controller.discover_and_activate()
        return controller

    def check_health(self) -> bool:
        """Checks the health of the dataprep service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ health check ] start to check health of milvus")
        try:
            if self.db_controller.active_component.check_health():
                if logflag:
                    logger.info("[ health check ] Successfully connected to Milvus!")
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
        """Ingest files/links content into milvus database.

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
            logger.info(f"[ milvus ingest ] files:{files}")
            logger.info(f"[ milvus ingest ] link_list:{link_list}")

        if files:
            if not isinstance(files, list):
                files = [files]

            for file in files:
                if logflag:
                    logger.info(f"[ milvus ingest ] processing file {file.filename}")

                # check whether the file already exists in milvus
                res = await self.db_controller.check_file_existance(file.filename)
                if res:
                    if logflag:
                        logger.info(f"[ milvus ingest] File {file.filename} already exist.")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded file {file.filename} already exists. Please change file name.",
                    )

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
                        logger.info(f"[ milvus ingest] Fail to ingest file {file.filename}")
                    raise HTTPException(status_code=400, detail=f"Fail to ingest {file.filename}. File already exists.")

                if logflag:
                    logger.info(f"[ milvus ingest] Successfully saved file {save_path}")

            result = {"status": 200, "message": "Data preparation succeeded"}
            return result

        if link_list:
            link_list = json.loads(link_list)  # Parse JSON string to list
            if not isinstance(link_list, list):
                raise HTTPException(status_code=400, detail=f"Link_list {link_list} should be a list.")
            for link in link_list:
                res = await self.db_controller.check_file_existance(link + ".txt")
                if res:
                    if logflag:
                        logger.info(f"[ redis ingest] Link {link} already exist.")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded link {link} already exists. Please change another link.",
                    )

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
                        logger.info(f"[ milvus ingest] Fail to ingest link {link}")
                    raise HTTPException(status_code=400, detail=f"Fail to ingest {link}. Link already exists.")

            if logflag:
                logger.info(f"[ milvus ingest] Successfully saved link list {link_list}")
            return {"status": 200, "message": "Data preparation succeeded"}

        raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

    async def get_files(self):
        """Get file structure from milvus database in the format of
        {
            "name": "File Name",
            "id": "File Name",
            "type": "File",
            "parent": "",
        }"""

        if logflag:
            logger.info("[ milvus get ] start to get file structure")

        file_list = await self.db_controller.get_file_list()
        return file_list

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file according to `file_path`.

        `file_path`:
            - specific file path (e.g. /path/to/file.txt)
            - "all": delete all files uploaded
        """
        if logflag:
            logger.info(f"[ milvus delete ] delete files: {file_path}")

        # delete all uploaded files
        if file_path == "all":
            if logflag:
                logger.info("[ milvus delete ] delete all files")

            res = await self.db_controller.delete_all_files()
            if not res:
                if logflag:
                    logger.info("[ milvus delete ] Fail to delete all files.")
                raise HTTPException(status_code=500, detail="Fail to delete all files.")

            # delete files on local disk
            try:
                remove_folder_with_ignore(upload_folder)
            except Exception as e:
                if logflag:
                    logger.info(f"[ milvus delete ] {e}. Fail to delete {upload_folder}.")
                raise HTTPException(status_code=500, detail=f"Fail to delete {upload_folder}.")

            if logflag:
                logger.info("[ milvus delete ] successfully delete all files.")

            create_upload_folder(upload_folder)
            if logflag:
                logger.info("[ milvus delete ] new upload folder created.")
            return {"status": True}

        # delete single file
        res = await self.db_controller.delete_single_file(file_path)
        if not res:
            if logflag:
                logger.info(f"[ milvus delete ] Fail to delete {file_path}.")
            raise HTTPException(status_code=404, detail=f"Fail to delete {file_path}.")

        delete_path = Path(upload_folder + "/" + encode_filename(file_path))
        if not delete_path.exists():
            if logflag:
                logger.info(f"[ redis delete ] File {file_path} not saved locally.")
            return {"status": True}

        # delete file
        if delete_path.is_file():
            delete_path.unlink()
            if logflag:
                logger.info(f"[ milvus delete ] file {file_path} deleted")
            return {"status": True}
        # folder
        else:
            if logflag:
                logger.info(f"[ milvus delete ] delete folder {file_path} is not supported for now.")
            raise HTTPException(status_code=404, detail=f"Delete folder {file_path} is not supported for now.")
