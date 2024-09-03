
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from config import COLLECTION_NAME, DISTANCE_STRATEGY, EMBED_MODEL, SEARCH_ENGINE, VDMS_HOST, VDMS_PORT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores.vdms import VDMS, VDMS_Client
from langchain_text_splitters import HTMLHeaderTextSplitter

from comps import DocPath, opea_microservices, opea_telemetry, register_microservice
from comps.dataprep.utils import document_loader, get_separators, get_tables_result

tei_embedding_endpoint = os.getenv("TEI_ENDPOINT")
client = VDMS_Client(VDMS_HOST, int(VDMS_PORT))


@register_microservice(
    name="opea_service@prepare_doc_vdms",
    endpoint="/v1/dataprep",
    host="0.0.0.0",
    port=6007,
    input_datatype=DocPath,
    output_datatype=None,
)
@opea_telemetry
def ingest_documents(doc_path: DocPath):
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

    content = document_loader(doc_path)
    chunks = text_splitter.split_text(content)
    if doc_path.process_table and path.endswith(".pdf"):
        table_chunks = get_tables_result(path, doc_path.table_strategy)
        chunks = chunks + table_chunks

    print("Done preprocessing. Created ", len(chunks), " chunks of the original pdf")

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

        _ = VDMS.from_texts(
            client=client,
            embedding=embedder,
            collection_name=COLLECTION_NAME,
            distance_strategy=DISTANCE_STRATEGY,
            engine=SEARCH_ENGINE,
            texts=batch_texts,
        )
        print(f"Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")


if __name__ == "__main__":
    opea_microservices["opea_service@prepare_doc_vdms"].start()