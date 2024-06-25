# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import uuid
from pathlib import Path
from typing import List, Optional, Union

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, INDEX_NAME, PG_CONNECTION_STRING
from fastapi import File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores import PGVector
from langsmith import traceable

from comps import DocPath, ServiceType, opea_microservices, register_microservice, register_statistics
from comps.dataprep.utils import document_loader, parse_html

tei_embedding_endpoint = os.getenv("TEI_ENDPOINT")


async def save_file_to_local_disk(save_path: str, file):
    save_path = Path(save_path)
    with save_path.open("wb") as fout:
        try:
            content = await file.read()
            fout.write(content)
        except Exception as e:
            print(f"Write file failed. Exception: {e}")
            raise HTTPException(status_code=500, detail=f"Write file {save_path} failed. Exception: {e}")


def ingest_doc_to_pgvector(doc_path: DocPath):
    """Ingest document to PGVector."""
    doc_path = doc_path.path
    print(f"Parsing document {doc_path}.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100, add_start_index=True)
    content = document_loader(doc_path)
    chunks = text_splitter.split_text(content)
    print("Done preprocessing. Created ", len(chunks), " chunks of the original pdf")
    print("PG Connection", PG_CONNECTION_STRING)

    # Create vectorstore
    if tei_embedding_endpoint:
        # create embeddings using TEI endpoint service
        embedder = HuggingFaceHubEmbeddings(model=tei_embedding_endpoint)
    else:
        # create embeddings using local embedding model
        embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    # Batch size
    batch_size = 32
    num_chunks = len(chunks)
    for i in range(0, num_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = batch_chunks

        _ = PGVector.from_texts(
            texts=batch_texts, embedding=embedder, collection_name=INDEX_NAME, connection_string=PG_CONNECTION_STRING
        )
        print(f"Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")
    return True


def ingest_link_to_pgvector(link_list: List[str]):
    data_collection = parse_html(link_list)

    texts = []
    metadatas = []
    for data, meta in data_collection:
        doc_id = str(uuid.uuid4())
        metadata = {"source": meta, "identify_id": doc_id}
        texts.append(data)
        metadatas.append(metadata)

    # Create vectorstore
    if tei_embedding_endpoint:
        # create embeddings using TEI endpoint service
        embedder = HuggingFaceHubEmbeddings(model=tei_embedding_endpoint)
    else:
        # create embeddings using local embedding model
        embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    _ = PGVector.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas,
        collection_name=INDEX_NAME,
        connection_string=PG_CONNECTION_STRING,
    )


@register_microservice(
    name="opea_service@prepare_doc_pgvector",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep",
    host="0.0.0.0",
    port=6007,
)
@traceable(run_type="tool")
@register_statistics(names=["opea_service@dataprep_pgvector"])
async def ingest_documents(
    files: Optional[Union[UploadFile, List[UploadFile]]] = File(None), link_list: Optional[str] = Form(None)
):
    print(f"files:{files}")
    print(f"link_list:{link_list}")
    if files and link_list:
        raise HTTPException(status_code=400, detail="Provide either a file or a string list, not both.")

    if files:
        if not isinstance(files, list):
            files = [files]
        upload_folder = "./uploaded_files/"
        if not os.path.exists(upload_folder):
            Path(upload_folder).mkdir(parents=True, exist_ok=True)
        for file in files:
            save_path = upload_folder + file.filename
            await save_file_to_local_disk(save_path, file)
            ingest_doc_to_pgvector(DocPath(path=save_path))
            print(f"Successfully saved file {save_path}")
        return {"status": 200, "message": "Data preparation succeeded"}

    if link_list:
        try:
            link_list = json.loads(link_list)  # Parse JSON string to list
            if not isinstance(link_list, list):
                raise HTTPException(status_code=400, detail="link_list should be a list.")
            ingest_link_to_pgvector(link_list)
            print(f"Successfully saved link list {link_list}")
            return {"status": 200, "message": "Data preparation succeeded"}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for link_list.")

    raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")


if __name__ == "__main__":
    opea_microservices["opea_service@prepare_doc_pgvector"].start()
