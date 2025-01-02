# Copyright (c) 2015-2024 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import lancedb
import msgpack
from config import (
    COLLECTION_NAME,
    LOCAL_EMBEDDING_MODEL,
    MINIO_ACCESS_KEY,
    MINIO_DOCUMENT_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    MINIO_WAREHOUSE_BUCKET,
    MOSEC_EMBEDDING_ENDPOINT,
    MOSEC_EMBEDDING_MODEL,
    TEI_EMBEDDING_ENDPOINT,
)
from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_text_splitters import HTMLHeaderTextSplitter
from minio import Minio, S3Error

from comps import CustomLogger, DocPath, opea_microservices, register_microservice
from comps.dataprep.minio.minio_schema import MinioEventNotification
from comps.dataprep.utils import (
    decode_filename,
    document_loader,
    encode_filename,
    get_separators,
    get_tables_result,
    parse_html,
)

logger = CustomLogger("prepare_doc_minio_lancedb")
logflag = os.getenv("LOGFLAG", True)

# workaround notes: cp comps/dataprep/utils.py ./lancedb/utils.py
# from utils import document_loader, get_tables_result, parse_html
INDEX_PARAMS = {"index_type": "FLAT", "metric_type": "IP", "params": {}}
PARTITION_FIELD_NAME = "filename"

minio_client = Minio(
    endpoint=MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE
)


class MosecEmbeddings(OpenAIEmbeddings):
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        _chunk_size = chunk_size or self.chunk_size
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


def ingest_chunks_to_lancedb(file_name: str, chunks: List):
    if logflag:
        logger.info(f"[ ingest chunks ] file name: {file_name}")

    tbl = my_lancedb.get_table()

    # insert documents to lancedb
    insert_text = []
    insert_metadata = []
    doc_ids = []
    for i, chunk in enumerate(chunks):
        insert_text.append(chunk)
        insert_metadata.append({PARTITION_FIELD_NAME: file_name})
        doc_ids.append(f"{file_name}_{i}")
    # Batch size
    batch_size = 32
    num_chunks = len(chunks)

    for i in range(0, num_chunks, batch_size):
        if logflag:
            logger.info(f"[ ingest chunks ] Current batch: {i}")
        batch_texts = insert_text[i : i + batch_size]
        batch_metadata = insert_metadata[i : i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch_texts)
        batch_doc_ids = doc_ids[i : i + batch_size]

        data_docs = []
        for j, doc in enumerate(batch_texts):
            data_docs.append(
                {"text": doc, "metadata": batch_metadata[j], "vector": batch_embeddings[j], "id": batch_doc_ids[j]}
            )

        try:

            if tbl is None:
                tbl = lancedb.connect("s3://warehouse/v-db").create_table(COLLECTION_NAME, data_docs)
            else:
                tbl.add(data_docs)
        except Exception as e:
            if logflag:
                logger.info(f"[ ingest chunks ] fail to ingest chunks into lancedb. error: {e}")
            raise HTTPException(status_code=500, detail=f"Fail to store chunks of file {file_name}.")

    if logflag:
        logger.info(f"[ ingest chunks ] Docs ingested file {file_name} to lancedb collection {COLLECTION_NAME}.")

    return True


def ingest_data_to_minio(doc_path: DocPath):
    """Ingest document to lancedb."""
    path = doc_path.path
    file_name = path.split("/")[-1]
    if logflag:
        logger.info(f"[ ingest data ] Parsing document {path}, file name: {file_name}.")

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

    content = document_loader(path)

    if logflag:
        logger.info("[ ingest data ] file content loaded")

    structured_types = [".xlsx", ".csv", ".json", "jsonl"]
    _, ext = os.path.splitext(path)

    if ext in structured_types:
        chunks = content
    else:
        chunks = text_splitter.split_text(content)

    if doc_path.process_table and path.endswith(".pdf"):
        table_chunks = get_tables_result(path, doc_path.table_strategy)
        chunks = chunks + table_chunks
    if logflag:
        logger.info(f"[ ingest data ] Done preprocessing. Created {len(chunks)} chunks of the original file.")

    return chunks


def search_by_file(collection, file_name):
    query = f"{PARTITION_FIELD_NAME} == '{file_name}'"
    results = collection.query(
        expr=query,
        output_fields=[PARTITION_FIELD_NAME, "pk"],
    )
    if logflag:
        logger.info(f"[ search by file ] searched by {file_name}")
        logger.info(f"[ search by file ] {len(results)} results: {results}")
    return results


def search_all(collection):
    results = collection.search(query="pk >= 0", output_fields=[PARTITION_FIELD_NAME, "pk"])
    if logflag:
        logger.info(f"[ search all ] {len(results)} results: {results}")
    return results


def delete_all_data():
    if logflag:
        logger.info("[ delete all ] deleting all data in lancedb")
    # List and delete all objects
    try:
        # Generate a list of all objects in the bucket
        objects = minio_client.list_objects(MINIO_DOCUMENT_BUCKET, recursive=True)

        # Delete each object
        for obj in objects:
            minio_client.remove_object(MINIO_DOCUMENT_BUCKET, obj.object_name)
            print(f"Deleted {obj.object_name}")

        print("All objects have been deleted from the bucket.")

    except S3Error as e:
        print("Error:", e)


def delete_by_partition_field(my_lancedb, partition_field):
    if logflag:
        logger.info(f"[ delete partition ] deleting {PARTITION_FIELD_NAME} {partition_field}")
    res = my_lancedb.delete(filter=f"metadata.{PARTITION_FIELD_NAME} == '{partition_field}'")
    if logflag:
        logger.info(f"[ delete partition ] delete success: {res}")


@register_microservice(
    name="opea_service@prepare_doc_minio_lancedb", endpoint="/v1/dataprep", host="0.0.0.0", port=6010
)
async def ingest_documents(
    files: Optional[Union[UploadFile, List[UploadFile]]] = File(None),
    link_list: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    process_table: bool = Form(False),
    table_strategy: str = Form("fast"),
):
    if logflag:
        logger.info(f"[ upload ] files:{files}")
        logger.info(f"[ upload ] link_list:{link_list}")

    if files and link_list:
        raise HTTPException(status_code=400, detail="Provide either a file or a string list, not both.")

    if files:
        if not isinstance(files, list):
            files = [files]
        uploaded_files = []

        for file in files:
            encode_file = encode_filename(file.filename)
            save_path = f"s3://{MINIO_DOCUMENT_BUCKET}/{encode_file}"
            if logflag:
                logger.info(f"[ upload ] processing file {save_path}")

            content = await file.read()
            file_size = len(content)
            file_data = io.BytesIO(content)

            minio_client.put_object(
                bucket_name=MINIO_DOCUMENT_BUCKET,
                object_name=encode_file,
                data=file_data,
                length=file_size,
                content_type=file.content_type,
                metadata={
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "process_table": process_table,
                    "table_strategy": table_strategy,
                },
            )

            uploaded_files.append(save_path)
            if logflag:
                logger.info(f"Saved file {save_path} into MinIO")

        results = {"status": 200, "message": "Data preparation succeeded"}
        if logflag:
            logger.info(results)
        return results

    if link_list:
        link_list = json.loads(link_list)  # Parse JSON string to list
        if not isinstance(link_list, list):
            raise HTTPException(status_code=400, detail="link_list should be a list.")

        for link in link_list:
            encoded_link = encode_filename(link)

            if logflag:
                logger.info(f"[ upload ] processing link {encoded_link}")

            encode_file = f"{encoded_link}.txt"
            content = parse_html([link])[0][0]
            file_size = len(content)
            file_data = io.BytesIO(content)

            minio_client.put_object(
                bucket_name=MINIO_DOCUMENT_BUCKET,
                object_name=encode_file,
                data=file_data,
                length=file_size,
                metadata={
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "process_table": process_table,
                    "table_strategy": table_strategy,
                },
            )

        if logflag:
            logger.info(f"[ upload ] Successfully saved link list {link_list}")
        return {"status": 200, "message": "Data preparation succeeded"}

    raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")


@register_microservice(
    name="opea_service@prepare_doc_minio_lancedb", endpoint="/v1/minio/document/notification", host="0.0.0.0", port=6010
)
async def process_documents(event: MinioEventNotification):
    # json_data = await request.json()
    # print(json.dumps(json_data, indent=2))
    print(event)
    if event.EventName == "s3:ObjectCreated:Put":
        for record in event.Records:
            bucket_name = record.s3.bucket.name
            object_name = record.s3.object.key
            _, file_extension = os.path.splitext(object_name)
            with tempfile.NamedTemporaryFile(delete=True, suffix=file_extension) as temp_file:
                temp_file_path = temp_file.name
                minio_client.fget_object(bucket_name, object_name, temp_file_path)
                chunks = ingest_data_to_minio(
                    DocPath(
                        path=temp_file_path,
                        chunk_size=record.s3.object.userMetadata.chunk_size,
                        chunk_overlap=record.s3.object.userMetadata.chunk_overlap,
                        process_table=record.s3.object.userMetadata.process_table,
                        table_strategy=record.s3.object.userMetadata.table_strategy,
                    )
                )
                msgpack_data = msgpack.packb(chunks)
                buffer = io.BytesIO(msgpack_data)
                buffer_size = buffer.getbuffer().nbytes
                minio_client.put_object(
                    MINIO_WAREHOUSE_BUCKET,
                    object_name=f"metadata/{object_name}.msgpack",
                    data=buffer,
                    length=buffer_size,
                    content_type="application/x-msgpack",
                )
    if event.EventName == "s3:ObjectRemoved:Delete":
        for record in event.Records:
            object_name = record.s3.object.key
            minio_client.remove_object(MINIO_WAREHOUSE_BUCKET, object_name=f"metadata/{object_name}.msgpack")
    return {"status": 200, "message": "Document processed successfully"}


@register_microservice(
    name="opea_service@prepare_doc_minio_lancedb", endpoint="/v1/minio/metadata/notification", host="0.0.0.0", port=6010
)
async def process_metadata(event: MinioEventNotification):
    # json_data = await request.json()
    # print(json.dumps(json_data, indent=2))
    if event.EventName == "s3:ObjectCreated:Put":
        for record in event.Records:
            bucket_name = record.s3.bucket.name
            object_name = record.s3.object.key
            response = minio_client.get_object(bucket_name, object_name)
            msgpack_data = response.read()
            response.close()
            response.release_conn()

            # Deserialize the MsgPack data back into a list
            chunk_list = msgpack.unpackb(msgpack_data)
            print(f"Total Chunks are {len(chunk_list)}")
            file_name = object_name.split(".msgpack")[0].split("metadata/")[1]
            ingest_chunks_to_lancedb(file_name, chunk_list)
    elif event.EventName == "s3:ObjectRemoved:Delete":
        # define lancedb obj
        for record in event.Records:
            object_name = record.s3.object.key
            file_name = object_name.split(".msgpack")[0].split("metadata/")[1]
            encode_file_name = encode_filename(file_name)
            try:
                delete_by_partition_field(my_lancedb, encode_file_name)
            except Exception as e:
                if logflag:
                    logger.info(f"[delete] fail to delete file {file_name}: {e}")
                return {"status": False}

    return {"status": 200, "message": "Metadata processed successfully"}


@register_microservice(
    name="opea_service@prepare_doc_minio_lancedb", endpoint="/v1/dataprep/get_file", host="0.0.0.0", port=6010
)
async def rag_get_file_structure():
    if logflag:
        logger.info("[ get ] start to get file structure")

    # collection does not exist
    if not my_lancedb:
        logger.info(f"[ get ] collection {COLLECTION_NAME} does not exist.")
        return []

    # get all files from db
    try:
        file_objects = minio_client.list_objects(MINIO_DOCUMENT_BUCKET)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed when searching in lancedb db for all files.")

    res_file = [res.object_name for res in file_objects]
    if logflag:
        logger.info(f"[ get ] unique list from db: {res_file}")

    # construct result file list in format
    file_list = []
    for file_name in res_file:
        file_dict = {
            "name": decode_filename(file_name),
            "id": decode_filename(file_name),
            "type": "File",
            "parent": "",
        }
        file_list.append(file_dict)

    if logflag:
        logger.info(f"[ get ] final file list: {file_list}")
    return file_list


@register_microservice(
    name="opea_service@prepare_doc_minio_lancedb", endpoint="/v1/dataprep/delete_file", host="0.0.0.0", port=6010
)
async def delete_single_file(file_path: str = Body(..., embed=True)):
    """Delete file according to `file_path`.

    `file_path`:
        - file/link path (e.g. /path/to/file.txt)
        - "all": delete all files uploaded
    """
    if logflag:
        logger.info(file_path)

    # delete all uploaded files
    if file_path == "all":
        if logflag:
            logger.info("[ delete ] deleting all files")

        delete_all_data()

        if logflag:
            logger.info("[ delete ] successfully delete all files.")

        return {"status": True}

    encode_file_name = encode_filename(file_path)

    if logflag:
        logger.info(f"[delete] deleting file {encode_file_name}")
    try:
        minio_client.remove_object(MINIO_DOCUMENT_BUCKET, encode_file_name)
    except Exception as e:
        if logflag:
            logger.info(f"[delete] fail to delete file {encode_file_name}: {e}")
        return {"status": False}
    if logflag:
        logger.info(f"[delete] file {file_path} deleted")
    return {"status": True}


if __name__ == "__main__":
    logger.info("[ prepare_doc_minio_lancedb ]  Using MinIO as the object storage.")
    # Create vectorstore
    if MOSEC_EMBEDDING_ENDPOINT:
        # create embeddings using MOSEC endpoint service
        if logflag:
            logger.info(
                f"[ prepare_doc_minio_lancedb ] MOSEC_EMBEDDING_ENDPOINT:{MOSEC_EMBEDDING_ENDPOINT}, MOSEC_EMBEDDING_MODEL:{MOSEC_EMBEDDING_MODEL}"
            )
        embeddings = MosecEmbeddings(model=MOSEC_EMBEDDING_MODEL)
    elif TEI_EMBEDDING_ENDPOINT:
        # create embeddings using TEI endpoint service
        if logflag:
            logger.info(f"[ prepare_doc_minio_lancedb ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
        embeddings = HuggingFaceHubEmbeddings(model=TEI_EMBEDDING_ENDPOINT)
    else:
        # create embeddings using local embedding model
        if logflag:
            logger.info(f"[ prepare_doc_minio_lancedb ] LOCAL_EMBEDDING_MODEL:{LOCAL_EMBEDDING_MODEL}")
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL, model_kwargs={"device": "cpu", "trust_remote_code": True}
        )
    # create lancedb
    my_lancedb = LanceDB(uri=f"s3://{MINIO_WAREHOUSE_BUCKET}/v-db", embedding=embeddings, table_name=COLLECTION_NAME)

    opea_microservices["opea_service@prepare_doc_minio_lancedb"].start()
    print("DOCPREP Server Started")
