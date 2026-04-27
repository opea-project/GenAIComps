# Copyright (C) 2026 Dnotitia
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import requests
from fastapi import Body, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter
from seahorse_vector_store import SeahorseVectorStore

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

logger = CustomLogger("seahorse_dataprep")
logflag = os.getenv("LOGFLAG", False)
upload_folder = "./uploaded_files/"
TEI_INFO_TIMEOUT_SECONDS = 10

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# TEI Embedding endpoint
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")
# Huggingface API token for TEI embedding endpoint
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Seahorse Cloud configuration
SEAHORSE_BASE_URL = os.getenv("SEAHORSE_BASE_URL", "")
SEAHORSE_API_KEY = os.getenv("SEAHORSE_API_KEY", "")
SEAHORSE_EMBEDDING_MODE = os.getenv("SEAHORSE_EMBEDDING_MODE", "builtin").strip().lower()


@OpeaComponentRegistry.register("OPEA_DATAPREP_SEAHORSE")
class OpeaSeahorseDataprep(OpeaComponent):
    """Seahorse Cloud document ingestion via API Gateway.

    Embedding mode (controlled by SEAHORSE_EMBEDDING_MODE env var):
    - "builtin": Seahorse Cloud generates embeddings server-side (no TEI needed).
                 Must match Retriever's builtin mode for consistent search.
    - "external": Uses TEI or local HuggingFace embeddings.
                  Must match Retriever's external mode for consistent search.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.upload_folder = upload_folder
        self.use_builtin = SEAHORSE_EMBEDDING_MODE == "builtin"
        self.embedder = self._initialize_embedder()
        self.vectorstore = self._initialize_vectorstore()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaSeahorseDataprep health check failed.")

    @staticmethod
    def _require_env_config() -> None:
        missing = []
        if not SEAHORSE_BASE_URL:
            missing.append("SEAHORSE_BASE_URL")
        if not SEAHORSE_API_KEY:
            missing.append("SEAHORSE_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required Seahorse configuration: {', '.join(missing)}")

    @staticmethod
    def _fetch_tei_model_id() -> str:
        try:
            response = requests.get(f"{TEI_EMBEDDING_ENDPOINT}/info", timeout=TEI_INFO_TIMEOUT_SECONDS)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available: {exc}",
            ) from exc

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available.",
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} returned invalid JSON.",
            ) from exc

        model_id = payload.get("model_id") if isinstance(payload, dict) else None
        if not model_id:
            raise HTTPException(
                status_code=400,
                detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} did not return model_id.",
            )
        return model_id

    def _initialize_embedder(self):
        if self.use_builtin:
            if logflag:
                logger.info("[ init embedder ] Using Seahorse built-in embeddings (SEAHORSE_EMBEDDING_MODE=builtin)")
            return None

        if TEI_EMBEDDING_ENDPOINT:
            if logflag:
                logger.info(f"[ init embedder ] TEI_EMBEDDING_ENDPOINT: {TEI_EMBEDDING_ENDPOINT}")
            if not HF_TOKEN:
                raise HTTPException(
                    status_code=400,
                    detail="You MUST offer the `HF_TOKEN` when using `TEI_EMBEDDING_ENDPOINT`.",
                )
            model_id = self._fetch_tei_model_id()
            return HuggingFaceInferenceAPIEmbeddings(
                api_key=HF_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        else:
            if logflag:
                logger.info(f"[ init embedder ] LOCAL EMBED_MODEL: {EMBED_MODEL}")

            return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    def _initialize_vectorstore(self) -> SeahorseVectorStore:
        self._require_env_config()
        kwargs = {
            "api_key": SEAHORSE_API_KEY,
            "base_url": SEAHORSE_BASE_URL,
        }
        if self.use_builtin:
            kwargs["use_builtin_embedding"] = True
        else:
            kwargs["embedding"] = self.embedder
            kwargs["use_builtin_embedding"] = False

        return SeahorseVectorStore(**kwargs)

    def check_health(self) -> bool:
        """Check Seahorse Cloud connectivity via the SDK's public health() API."""
        if logflag:
            logger.info("[ check health ] start to check health of Seahorse Cloud")
        try:
            self.vectorstore.health()
            if logflag:
                logger.info("[ check health ] Seahorse Cloud connected.")
            return True
        except Exception as e:
            logger.error(f"[ check health ] Seahorse Cloud health check failed: {e}")
            return False

    def invoke(self, *args, **kwargs):
        pass

    async def ingest_data_to_seahorse(self, doc_path: DocPath):
        """Parse, chunk, and ingest a single document."""
        path = doc_path.path
        file_name = path.split("/")[-1]
        if logflag:
            logger.info(f"[ ingest ] Parsing document {path}")

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
            if table_chunks:
                chunks = chunks + table_chunks

        if logflag:
            logger.info(f"[ ingest ] Created {len(chunks)} chunks from {file_name}")

        metadatas = [{"filename": file_name} for _ in chunks]

        # builtin: text only → server generates embeddings
        # external: SDK uses self.embedder to generate embeddings client-side
        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)

        if logflag:
            logger.info(f"[ ingest ] Successfully ingested {file_name} to Seahorse Cloud")
        return True

    async def ingest_files(self, input: DataprepRequest):
        """Ingest files/links into Seahorse Cloud.

        Args:
            input (DataprepRequest): files, link_list, chunk_size, chunk_overlap, etc.
        Returns:
            dict: {"status": 200, "message": "Data preparation succeeded"}
        """
        files = input.files
        link_list = input.link_list
        chunk_size = input.chunk_size
        chunk_overlap = input.chunk_overlap
        process_table = input.process_table
        table_strategy = input.table_strategy

        if logflag:
            logger.info(f"[ ingest ] files: {files}")
            logger.info(f"[ ingest ] link_list: {link_list}")

        if files:
            if not isinstance(files, list):
                files = [files]
            for file in files:
                encode_file = encode_filename(file.filename)
                save_path = self.upload_folder + encode_file
                await save_content_to_local_disk(save_path, file)
                await self.ingest_data_to_seahorse(
                    DocPath(
                        path=save_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        process_table=process_table,
                        table_strategy=table_strategy,
                    )
                )
                if logflag:
                    logger.info(f"[ ingest ] Successfully saved file {save_path}")
            return {"status": 200, "message": "Data preparation succeeded"}

        if link_list:
            link_list = json.loads(link_list)
            if not isinstance(link_list, list):
                raise HTTPException(status_code=400, detail="link_list should be a list.")
            for link in link_list:
                encoded_link = encode_filename(link)
                save_path = self.upload_folder + encoded_link + ".txt"
                content = parse_html_new([link], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                await save_content_to_local_disk(save_path, content)
                await self.ingest_data_to_seahorse(
                    DocPath(
                        path=save_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        process_table=process_table,
                        table_strategy=table_strategy,
                    )
                )
                if logflag:
                    logger.info(f"[ ingest ] Successfully saved link {link}")
            return {"status": 200, "message": "Data preparation succeeded"}

        raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

    async def get_files(self):
        """Get list of ingested files from local upload folder."""
        if logflag:
            logger.info("[ get files ] start to get file structure")

        if not Path(self.upload_folder).exists():
            if logflag:
                logger.info("No file uploaded, return empty list.")
            return []

        file_content = get_file_structure(self.upload_folder)
        if logflag:
            logger.info(file_content)
        return file_content

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file data from Seahorse Cloud.

        file_path:
        - "all": delete all data from Seahorse Cloud and local upload folder
        - specific path: delete chunks for that file (by metadata filter)
        """
        if logflag:
            logger.info(f"[ delete ] file_path: {file_path}")

        if file_path == "all":
            self.vectorstore.delete(delete_all=True)
            if logflag:
                logger.info("[ delete ] successfully deleted all data from Seahorse Cloud")
            try:
                remove_folder_with_ignore(self.upload_folder)
            except Exception as e:
                logger.error(f"[ delete ] Failed to remove upload folder: {e}")
            create_upload_folder(self.upload_folder)
            if logflag:
                logger.info("[ delete ] successfully deleted all local files")
            return {"status": True}

        encode_file_name = encode_filename(file_path)
        delete_path = Path(self.upload_folder + "/" + encode_file_name)

        if delete_path.exists():
            self.vectorstore.delete(filter={"filename": encode_file_name})
            delete_path.unlink()
            if logflag:
                logger.info(
                    f"[ delete ] file {file_path} (stored as {encode_file_name}) deleted from Seahorse Cloud and local"
                )
            return {"status": True}
        else:
            raise HTTPException(status_code=404, detail="File not found.")
