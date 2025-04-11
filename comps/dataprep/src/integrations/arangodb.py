# Copyright (C) 2025 ArangoDB Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import List, Optional, Union

import openai
from arango import ArangoClient
from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_arangodb import ArangoGraph
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.dataprep.src.utils import (
    decode_filename,
    document_loader,
    encode_filename,
    get_separators,
    get_tables_result,
    parse_html,
    save_content_to_local_disk,
)

logger = CustomLogger("OPEA_DATAPREP_ARANGODB")
logflag = os.getenv("LOGFLAG", False)


# ArangoDB configuration
ARANGO_URL = os.getenv("ARANGO_URL", "http://localhost:8529")
ARANGO_DB_NAME = os.getenv("ARANGO_DB_NAME", "_system")
ARANGO_USERNAME = os.getenv("ARANGO_USERNAME", "root")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD", "test")

# ArangoDB graph configuration
ARANGO_INSERT_ASYNC = os.getenv("ARANGO_INSERT_ASYNC", False)
ARANGO_BATCH_SIZE = os.getenv("ARANGO_BATCH_SIZE", 1000)
ARANGO_USE_GRAPH_NAME = os.getenv("ARANGO_USE_GRAPH_NAME", True)
ARANGO_GRAPH_NAME = os.getenv("ARANGO_GRAPH_NAME", "GRAPH")

# VLLM configuration
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:80")
VLLM_MODEL_ID = os.getenv("VLLM_MODEL_ID", "Intel/neural-chat-7b-v3-3")
VLLM_MAX_NEW_TOKENS = os.getenv("VLLM_MAX_NEW_TOKENS", 512)
VLLM_TOP_P = os.getenv("VLLM_TOP_P", 0.9)
VLLM_TEMPERATURE = os.getenv("VLLM_TEMPERATURE", 0.8)
VLLM_TIMEOUT = os.getenv("VLLM_TIMEOUT", 600)

# TEI configuration
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT")
TEI_EMBED_MODEL = os.getenv("TEI_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBED_SOURCE_DOCUMENTS = os.getenv("EMBED_SOURCE_DOCUMENTS", "true").lower() == "true"
EMBED_NODES = os.getenv("EMBED_NODES", "true").lower() == "true"
EMBED_RELATIONSHIPS = os.getenv("EMBED_RELATIONSHIPS", "true").lower() == "true"

# OpenAI configuration (alternative to TEI/VLLM)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_EMBED_DIMENSION = os.getenv("OPENAI_EMBED_DIMENSION", 512)
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_CHAT_TEMPERATURE = os.getenv("OPENAI_CHAT_TEMPERATURE", 0)
OPENAI_CHAT_ENABLED = os.getenv("OPENAI_CHAT_ENABLED", "true").lower() == "true"
OPENAI_EMBED_ENABLED = os.getenv("OPENAI_EMBED_ENABLED", "true").lower() == "true"

# LLM/Graph Transformer configuration
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH")
ALLOWED_NODES = os.getenv("ALLOWED_NODES", [])
ALLOWED_RELATIONSHIPS = os.getenv("ALLOWED_RELATIONSHIPS", [])
NODE_PROPERTIES = os.getenv("NODE_PROPERTIES", ["description"])
RELATIONSHIP_PROPERTIES = os.getenv("RELATIONSHIP_PROPERTIES", ["description"])
ENTITY_CAPITALIZATION_STRATEGY = os.getenv("ENTITY_CAPITALIZATION_STRATEGY", "upper")
INCLUDE_SOURCE = os.getenv("INCLUDE_SOURCE", "true").lower() == "true"


@OpeaComponentRegistry.register("OPEA_DATAPREP_ARANGODB")
class OpeaArangoDataprep(OpeaComponent):
    """Dataprep component for ArangoDB ingestion and search services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.upload_folder = "./uploaded_files/"

        self.llm_transformer: LLMGraphTransformer
        self.embeddings: Embeddings

        self._initialize_llm()
        self._initialize_embeddings()
        self._initialize_client()

        if not self.check_health():
            logger.error("OpeaArangoDataprep health check failed.")

    def _initialize_llm(self):
        """Initialize the LLM model & LLMGraphTransformer object."""

        allowed_nodes = ALLOWED_NODES
        allowed_relationships = ALLOWED_RELATIONSHIPS
        node_properties = NODE_PROPERTIES
        relationship_properties = RELATIONSHIP_PROPERTIES

        # Process string inputs if needed
        if allowed_nodes and isinstance(allowed_nodes, str):
            allowed_nodes = allowed_nodes.split(",")

        if allowed_relationships and isinstance(allowed_relationships, str):
            allowed_relationships = allowed_relationships.split(",")

        if node_properties and isinstance(node_properties, str):
            node_properties = node_properties.split(",")

        if relationship_properties and isinstance(relationship_properties, str):
            relationship_properties = relationship_properties.split(",")

        prompt_template = None
        if SYSTEM_PROMPT_PATH is not None:
            try:
                with open(SYSTEM_PROMPT_PATH, "r") as f:
                    prompt_template = ChatPromptTemplate.from_messages(
                        [
                            ("system", f.read()),
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
            except Exception as e:
                logger.error(f"Could not set custom Prompt: {e}")

        ignore_tool_usage = False

        if OPENAI_API_KEY and OPENAI_CHAT_ENABLED:
            if logflag:
                logger.info("OpenAI API Key is set. Verifying its validity...")
            openai.api_key = OPENAI_API_KEY

            try:
                openai.models.list()
                if logflag:
                    logger.info("OpenAI API Key is valid.")
                llm = ChatOpenAI(temperature=OPENAI_CHAT_TEMPERATURE, model_name=OPENAI_CHAT_MODEL)
            except openai.error.AuthenticationError:
                if logflag:
                    logger.info("OpenAI API Key is invalid.")
            except Exception as e:
                if logflag:
                    logger.info(f"An error occurred while verifying the API Key: {e}")
        elif VLLM_ENDPOINT:
            llm = ChatOpenAI(
                openai_api_key=VLLM_API_KEY,
                openai_api_base=f"{VLLM_ENDPOINT}/v1",
                model=VLLM_MODEL_ID,
                temperature=VLLM_TEMPERATURE,
                max_tokens=VLLM_MAX_NEW_TOKENS,
                top_p=VLLM_TOP_P,
                timeout=VLLM_TIMEOUT,
            )
            ignore_tool_usage = True
        else:
            raise HTTPException(status_code=400, detail="No LLM environment variables are set, cannot generate graphs.")

        try:
            self.llm_transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                prompt=prompt_template,
                node_properties=node_properties or False,
                relationship_properties=relationship_properties or False,
                ignore_tool_usage=ignore_tool_usage,
            )
        except (TypeError, ValueError) as e:
            if logflag:
                logger.warning(f"Advanced LLMGraphTransformer failed: {e}")
            # Fall back to basic config
            try:
                self.llm_transformer = LLMGraphTransformer(llm=llm, ignore_tool_usage=ignore_tool_usage)
            except (TypeError, ValueError) as e:
                if logflag:
                    logger.error(f"Failed to initialize LLMGraphTransformer: {e}")

                raise HTTPException(status_code=500, detail=f"Failed to initialize LLMGraphTransformer: {e}")

    def _initialize_embeddings(self):
        """Initialize the embeddings model."""

        if TEI_EMBEDDING_ENDPOINT and HUGGINGFACEHUB_API_TOKEN:
            self.embeddings = HuggingFaceHubEmbeddings(
                model=TEI_EMBEDDING_ENDPOINT,
                huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            )
        elif TEI_EMBED_MODEL:
            self.embeddings = HuggingFaceBgeEmbeddings(model_name=TEI_EMBED_MODEL)
        else:
            raise HTTPException(
                status_code=400, detail="No embeddings environment variables are set, cannot generate embeddings."
            )

    def _initialize_client(self):
        """Initialize the ArangoDB connection."""

        self.client = ArangoClient(hosts=ARANGO_URL)
        sys_db = self.client.db(name="_system", username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)

        if not sys_db.has_database(ARANGO_DB_NAME):
            sys_db.create_database(ARANGO_DB_NAME)

        self.db = self.client.db(name=ARANGO_DB_NAME, username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)
        if logflag:
            logger.info(f"Connected to ArangoDB {self.db.version()}.")

    def check_health(self) -> bool:
        """Checks the health of the retriever service."""

        if logflag:
            logger.info("[ check health ] start to check health of ArangoDB")
        try:
            version = self.db.version()
            if logflag:
                logger.info(f"[ check health ] Successfully connected to ArangoDB {version}!")
            return True
        except Exception as e:
            logger.info(f"[ check health ] Failed to connect to ArangoDB: {e}")
            return False

    def ingest_data_to_arango(self, doc_path: DocPath):
        """Ingest document to ArangoDB."""

        path = doc_path.path
        if logflag:
            logger.info(f"Parsing document {path}")

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
            logger.info(f"Created {len(chunks)} chunks of the original file")

        ################################
        # Graph generation & insertion #
        ################################

        if ARANGO_USE_GRAPH_NAME:
            graph_name = ARANGO_GRAPH_NAME
        else:
            file_name = os.path.basename(path).split(".")[0]
            graph_name = "".join(c for c in file_name if c.isalnum() or c in "_-:.@()+,=;$!*'%")

        if logflag:
            logger.info(f"Creating graph {graph_name}.")

        graph = ArangoGraph(db=self.db, generate_schema_on_init=False)

        for i, text in enumerate(chunks):
            document = Document(page_content=text, metadata={"file_name": path, "chunk_index": i})

            if logflag:
                logger.info(f"Chunk {i}: extracting nodes & relationships")

            graph_doc = self.llm_transformer.process_response(document)

            if logflag:
                logger.info(f"Chunk {i}: inserting into ArangoDB")

            graph.add_graph_documents(
                graph_documents=[graph_doc],
                include_source=INCLUDE_SOURCE,
                graph_name=graph_name,
                update_graph_definition_if_exists=False,
                batch_size=ARANGO_BATCH_SIZE,
                use_one_entity_collection=True,
                insert_async=ARANGO_INSERT_ASYNC,
                embeddings=self.embeddings,
                embedding_field="embedding",
                embed_source=EMBED_SOURCE_DOCUMENTS,
                embed_nodes=EMBED_NODES,
                embed_relationships=EMBED_RELATIONSHIPS,
                capitalization_strategy=ENTITY_CAPITALIZATION_STRATEGY,
            )

            if logflag:
                logger.info(f"Chunk {i}: processed")

        if logflag:
            logger.info(f"Graph {graph_name} created with {len(chunks)} chunks.")

        return graph_name

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
        """Ingest files/links content into ArangoDB database.

        Save in the format of vector[768].
        Returns '{"status": 200, "message": "Data preparation succeeded"}' if successful.
        Args:
            files (Union[UploadFile, List[UploadFile]], optional): A file or a list of files to be ingested. Defaults to File(None).
            link_list (str, optional): A list of links to be ingested. Defaults to Form(None).
            chunk_size (int, optional): The size of the chunks to be split. Defaults to Form(500).
            chunk_overlap (int, optional): The overlap between chunks. Defaults to Form(100).
            process_table (bool, optional): Whether to process tables in PDFs. Defaults to Form(False).
            table_strategy (str, optional): The strategy to process tables in PDFs. Defaults to Form("fast").
        """

        if logflag:
            logger.info(f"files:{files}")
            logger.info(f"link_list:{link_list}")

        if not files and not link_list:
            raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

        graph_names_created = set()

        if files:
            if not isinstance(files, list):
                files = [files]
            uploaded_files = []
            for file in files:
                encode_file = encode_filename(file.filename)
                save_path = self.upload_folder + encode_file
                await save_content_to_local_disk(save_path, file)
                try:
                    graph_name = self.ingest_data_to_arango(
                        DocPath(
                            path=save_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            process_table=process_table,
                            table_strategy=table_strategy,
                        ),
                    )

                    uploaded_files.append(save_path)
                    graph_names_created.add(graph_name)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to ingest {save_path} into ArangoDB: {e}")

                if logflag:
                    logger.info(f"Successfully saved file {save_path}")

        if link_list:
            link_list = json.loads(link_list)  # Parse JSON string to list
            if not isinstance(link_list, list):
                raise HTTPException(status_code=400, detail="link_list should be a list.")
            for link in link_list:
                encoded_link = encode_filename(link)
                save_path = self.upload_folder + encoded_link + ".txt"
                content = parse_html([link])[0][0]
                await save_content_to_local_disk(save_path, content)
                try:
                    graph_name = self.ingest_data_to_arango(
                        DocPath(
                            path=save_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            process_table=process_table,
                            table_strategy=table_strategy,
                        ),
                    )
                    graph_names_created.add(graph_name)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to ingest {save_path} into ArangoDB: {e}")

                if logflag:
                    logger.info(f"Successfully saved link {link}")

        graph_names_created = list(graph_names_created)

        result = {
            "status": 200,
            "message": f"Data preparation succeeded: {graph_names_created}",
            "graph_names": graph_names_created,
        }

        if logflag:
            logger.info(result)

        return result

    def invoke(self, *args, **kwargs):
        pass

    async def get_files(self):
        """Get file structure from ArangoDB in the format of
        {
            "name": "File Name",
            "id": "File Name",
            "graph": "Graph Name",
            "type": "File",
            "parent": "",
        }"""

        res_list = []

        for graph in self.db.graphs():
            source_collection = f"{graph['name']}_SOURCE"

            query = f"""
                FOR chunk IN @@source_collection
                    COLLECT file_name = chunk.file_name
                    RETURN file_name
            """

            cursor = self.db.aql.execute(query, bind_vars={"@source_collection": source_collection})

            for file_name in cursor:
                res_list.append(
                    {
                        "name": decode_filename(file_name),
                        "id": decode_filename(file_name),
                        "graph": graph["name"],
                        "type": "File",
                        "parent": "",
                    }
                )

        if logflag:
            logger.info(f"[ arango get ] number of files: {len(res_list)}")

        return res_list

    async def delete_files(self, file_path: str = Body(..., embed=True)):
        """Delete file according to `file_path`.

        `file_path`:
            - specific file path (e.g. /path/to/file.txt)
            - "all": delete all files uploaded
        """

        if file_path == "all":
            for graph in self.db.graphs():
                self.db.delete_graph(graph["name"], drop_collections=True)
        else:
            if ARANGO_USE_GRAPH_NAME:
                self.db.delete_graph(ARANGO_GRAPH_NAME, drop_collections=True)
            else:
                file_name = os.path.basename(file_path).split(".")[0]
                graph_name = "".join(c for c in file_name if c.isalnum() or c in "_-:.@()+,=;$!*'%")

                self.db.delete_graph(graph_name, drop_collections=True)
