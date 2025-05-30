# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import List, Optional, Union

import openai
from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import HTMLHeaderTextSplitter

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DataprepRequest
from comps.dataprep.src.utils import (
    document_loader,
    encode_filename,
    get_separators,
    get_tables_result,
    parse_html,
    save_content_to_local_disk,
)

logger = CustomLogger("opea_dataprep_neo4j_langchain")
logflag = os.getenv("LOGFLAG", False)


# Neo4J configuration
NEO4J_URL = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")

# LLM endpoints
TGI_LLM_ENDPOINT = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
TGI_LLM_ENDPOINT_NO_RAG = os.getenv("TGI_LLM_ENDPOINT_NO_RAG", "http://localhost:8081")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


@OpeaComponentRegistry.register("OPEA_DATAPREP_NEO4J_LANGCHAIN")
class OpeaNeo4jDataprep(OpeaComponent):
    """Dataprep component for Neo4j ingestion and search services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.upload_folder = "./uploaded_files/"

        if OPENAI_KEY:
            logger.info("OpenAI API Key is set. Verifying its validity...")
            openai.api_key = OPENAI_KEY

            try:
                response = openai.Engine.list()
                logger.info("OpenAI API Key is valid.")
                llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
            except openai.error.AuthenticationError:
                logger.info("OpenAI API Key is invalid.")
            except Exception as e:
                logger.info(f"An error occurred while verifying the API Key: {e}")
        else:
            llm = HuggingFaceEndpoint(
                endpoint_url=TGI_LLM_ENDPOINT,
                max_new_tokens=512,
                top_k=40,
                top_p=0.9,
                temperature=0.8,
                timeout=600,
            )

        self.llm_transformer = LLMGraphTransformer(
            llm=llm, node_properties=["description"], relationship_properties=["description"]
        )
        self.graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

        # Perform health check
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaNeo4jDataprep health check failed.")

    def check_health(self) -> bool:
        """Checks the health of the Neo4j service."""
        if self.graph is None:
            logger.error("Neo4j graph is not initialized.")
            return False

        return True

    def invoke(self, *args, **kwargs):
        pass

    async def ingest_data_to_neo4j(self, doc_path: DocPath):
        """Ingest document to Neo4J."""
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
            logger.info("Done preprocessing. Created ", len(chunks), " chunks of the original file.")

        doc_list = [Document(page_content=text) for text in chunks]
        graph_doc = self.llm_transformer.convert_to_graph_documents(doc_list)
        self.graph.add_graph_documents(graph_doc, baseEntityLabel=True, include_source=True)

        if logflag:
            logger.info("The graph is built.")

        return True

    async def ingest_files(self, input: DataprepRequest):
        """Ingest files/links content into Neo4j database.

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
                await self.ingest_data_to_neo4j(
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
            link_list = json.loads(link_list)  # Parse JSON string to list
            if not isinstance(link_list, list):
                raise HTTPException(status_code=400, detail="link_list should be a list.")
            for link in link_list:
                encoded_link = encode_filename(link)
                save_path = self.upload_folder + encoded_link + ".txt"
                content = parse_html([link])[0][0]
                try:
                    await save_content_to_local_disk(save_path, content)
                    await self.ingest_data_to_neo4j(
                        DocPath(
                            path=save_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            process_table=process_table,
                            table_strategy=table_strategy,
                        )
                    )
                except json.JSONDecodeError:
                    raise HTTPException(status_code=500, detail="Fail to ingest data into Neo4j.")

                if logflag:
                    logger.info(f"Successfully saved link {link}")

            result = {"status": 200, "message": "Data preparation succeeded"}
            if logflag:
                logger.info(result)
            return result

        raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

    async def get_files(self):
        """Get file structure from pipecone database in the format of
        {
            "name": "File Name",
            "id": "File Name",
            "type": "File",
            "parent": "",
        }"""
        pass

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file according to `file_path`.

        `file_path`:
            - specific file path (e.g. /path/to/file.txt)
            - "all": delete all files uploaded
        """
        pass
