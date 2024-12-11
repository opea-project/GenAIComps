# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import List, Optional, Union

import openai
from arango import ArangoClient
from config import (
    ARANGO_DB_NAME,
    ARANGO_PASSWORD,
    ARANGO_URL,
    ARANGO_USERNAME,
    HUGGINGFACEHUB_API_TOKEN,
    OPENAI_API_KEY,
    TEI_EMBED_MODEL,
    TEI_EMBEDDING_ENDPOINT,
    TGI_LLM_ENDPOINT,
    OPENAI_EMBED_MODEL,
    OPENAI_EMBED_DIMENSIONS,
    USE_ONE_ENTITY_COLLECTION,
    INSERT_ASYNC,
    ARANGO_BATCH_SIZE,
    INCLUDE_SOURCE,
    SYSTEM_PROMPT_PATH,
    ALLOWED_NODES,
    ALLOWED_RELATIONSHIPS,
    NODE_PROPERTIES,
    RELATIONSHIP_PROPERTIES,
)
from fastapi import File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.graphs.arangodb_graph import ArangoGraph
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_core.prompts import ChatPromptTemplate, BasePromptTemplate

from comps import CustomLogger, DocPath, opea_microservices, register_microservice
from comps.dataprep.utils import (
    document_loader,
    encode_filename,
    get_separators,
    get_tables_result,
    parse_html,
    save_content_to_local_disk,
)

logger = CustomLogger("prepare_doc_arango")
logflag = os.getenv("LOGFLAG", False)

upload_folder = "./uploaded_files/"

PROMPT_TEMPLATE = None
if SYSTEM_PROMPT_PATH is not None:
    try:
        with open(SYSTEM_PROMPT_PATH, "r") as f:
            PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f.read(),
                    ),
                    (
                        "human",
                        (
                            "Tip: Make sure to answer in the correct format and do "
                            "not include any explanations. "
                            "Use the given format to extract information from the "
                            "following input: {input}"
                        ),
                    ),
                ]
            )
    except Exception:
        print("Could not set custom Prompt")

def ingest_data_to_arango(doc_path: DocPath, graph_name: str, create_embeddings: bool) -> bool:
    """Ingest document to ArangoDB."""
    path = doc_path.path
    if logflag:
        logger.info(f"Parsing document {path}.")

    ############
    # ArangoDB #
    ############

    client = ArangoClient(hosts=ARANGO_URL)
    sys_db = client.db(name="_system", username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)

    if not sys_db.has_database(ARANGO_DB_NAME):
        sys_db.create_database(ARANGO_DB_NAME)

    db = client.db(name=ARANGO_DB_NAME, username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)

    graph = ArangoGraph(
        db=db,
        include_examples=False,
        generate_schema_on_init=False,
    )

    #############################
    # Text Generation Inference #
    #############################

    if OPENAI_API_KEY:
        logger.info("OpenAI API Key is set. Verifying its validity...")
        openai.api_key = OPENAI_API_KEY

        try:
            openai.models.list()
            logger.info("OpenAI API Key is valid.")
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        except openai.error.AuthenticationError:
            logger.info("OpenAI API Key is invalid.")
        except Exception as e:
            logger.info(f"An error occurred while verifying the API Key: {e}")

    elif TGI_LLM_ENDPOINT:
        llm = HuggingFaceEndpoint(
            endpoint_url=TGI_LLM_ENDPOINT,
            max_new_tokens=512,
            top_k=40,
            top_p=0.9,
            temperature=0.8,
            timeout=600,
        )
    else:
        raise ValueError("No text generation inference endpoint is set.")

    try:
        if not (NODE_PROPERTIES or RELATIONSHIP_PROPERTIES):
            llm_transformer = LLMGraphTransformer(
                llm=llm, 
                prompt=PROMPT_TEMPLATE,
                allowed_nodes=ALLOWED_NODES,
                allowed_relationships=ALLOWED_RELATIONSHIPS,
            )
        else:
            llm_transformer = LLMGraphTransformer(
                llm=llm, 
                node_properties=NODE_PROPERTIES,
                relationship_properties=RELATIONSHIP_PROPERTIES,
                prompt=PROMPT_TEMPLATE,
                allowed_nodes=ALLOWED_NODES,
                allowed_relationships=ALLOWED_RELATIONSHIPS,
            )
    except (TypeError, ValueError) as e:
        logger.warning(f"Advanced LLMGraphTransformer failed: {e}")
        # Fall back to basic config
        try:
            llm_transformer = LLMGraphTransformer(llm=llm)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to initialize LLMGraphTransformer: {e}")
            raise

    ########################################
    # Text Embeddings Inference (optional) #
    ########################################

    if create_embeddings:
        if OPENAI_API_KEY:
            # Use OpenAI embeddings
            embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBED_MODEL,
                dimensions=OPENAI_EMBED_DIMENSIONS,
            )

        elif TEI_EMBEDDING_ENDPOINT and HUGGINGFACEHUB_API_TOKEN:
            # Use TEI endpoint service
            embeddings = HuggingFaceHubEmbeddings(
                model=TEI_EMBEDDING_ENDPOINT,
                huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            )
        elif TEI_EMBED_MODEL:
            # Use local embedding model
            embeddings = HuggingFaceBgeEmbeddings(model_name=TEI_EMBED_MODEL)
        else:
            logger.error("No text embeddings inference endpoint is set.")
            embeddings = None
    else:
        embeddings = None

    ############
    # Chunking #
    ############

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

    structured_types = [".xlsx", ".csv", ".json", "jsonl"]
    _, ext = os.path.splitext(path)

    if ext in structured_types:
        chunks = content
    else:
        chunks = text_splitter.split_text(content)

    if doc_path.process_table and path.endswith(".pdf"):
        table_chunks = get_tables_result(path, doc_path.table_strategy)
        if isinstance(table_chunks, list):
            chunks = chunks + table_chunks
    if logflag:
        logger.info("Done preprocessing. Created ", len(chunks), " chunks of the original file.")

    ################################
    # Graph generation & insertion #
    ################################

    generate_chunk_embeddings = embeddings is not None

    for text in chunks:
        document = Document(page_content=text)
        graph_docs = llm_transformer.convert_to_graph_documents([document])

        if generate_chunk_embeddings:
            source = graph_docs[0].source
            source.metadata["embeddings"] = embeddings.embed_documents([source.page_content])[0]

        graph.add_graph_documents(
            graph_documents=graph_docs,
            include_source=INCLUDE_SOURCE,
            graph_name=graph_name,
            update_graph_definition_if_exists=not USE_ONE_ENTITY_COLLECTION,
            batch_size=ARANGO_BATCH_SIZE,
            use_one_entity_collection=USE_ONE_ENTITY_COLLECTION, 
            insert_async=INSERT_ASYNC,
        )

    if logflag:
        logger.info("The graph is built.")

    return True


@register_microservice(
    name="opea_service@prepare_doc_arango",
    endpoint="/v1/dataprep",
    host="0.0.0.0",
    port=6007,
    input_datatype=DocPath,
    output_datatype=None,
)
async def ingest_documents(
    files: Optional[Union[UploadFile, List[UploadFile]]] = File(None),
    link_list: Optional[str] = Form(None),
    chunk_size: int = Form(1500),
    chunk_overlap: int = Form(100),
    process_table: bool = Form(False),
    table_strategy: str = Form("fast"),
    graph_name: str = Form("NewGraph"),
    create_embeddings: bool = Form(True),
):
    if logflag:
        logger.info(f"files:{files}")
        logger.info(f"link_list:{link_list}")

    if files:
        if not isinstance(files, list):
            files = [files]
        uploaded_files = []
        for file in files:
            encode_file = encode_filename(file.filename)
            save_path = upload_folder + encode_file
            await save_content_to_local_disk(save_path, file)
            ingest_data_to_arango(
                DocPath(
                    path=save_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    process_table=process_table,
                    table_strategy=table_strategy,
                ),
                graph_name=graph_name,
                create_embeddings=create_embeddings,
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
            save_path = upload_folder + encoded_link + ".txt"
            content = parse_html([link])[0][0]
            try:
                await save_content_to_local_disk(save_path, content)
                ingest_data_to_arango(
                    DocPath(
                        path=save_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        process_table=process_table,
                        table_strategy=table_strategy,
                    ),
                    graph_name=graph_name,
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


if __name__ == "__main__":
    opea_microservices["opea_service@prepare_doc_arango"].start()
