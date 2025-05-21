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
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter

from comps import CustomLogger, DocPath, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import ArangoDBDataprepRequest, DataprepRequest
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
EMBED_NODES = os.getenv("EMBED_NODES", "true").lower() == "true"
EMBED_EDGES = os.getenv("EMBED_EDGES", "true").lower() == "true"
EMBED_CHUNKS = os.getenv("EMBED_CHUNKS", "true").lower() == "true"

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
ALLOWED_NODE_TYPES = os.getenv("ALLOWED_NODE_TYPES", [])
ALLOWED_EDGE_TYPES = os.getenv("ALLOWED_EDGE_TYPES", [])
NODE_PROPERTIES = os.getenv("NODE_PROPERTIES", ["description"])
EDGE_PROPERTIES = os.getenv("EDGE_PROPERTIES", ["description"])
TEXT_CAPITALIZATION_STRATEGY = os.getenv("TEXT_CAPITALIZATION_STRATEGY", "upper")
INCLUDE_CHUNKS = os.getenv("INCLUDE_CHUNKS", "true").lower() == "true"


@OpeaComponentRegistry.register("OPEA_DATAPREP_ARANGODB")
class OpeaArangoDataprep(OpeaComponent):
    """Dataprep component for ArangoDB ingestion and search services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.upload_folder = "./uploaded_files/"

        self.llm_transformer: LLMGraphTransformer
        self.embeddings: Embeddings

        self._initialize_embeddings()
        self._initialize_client()

        if not self.check_health():
            logger.error("OpeaArangoDataprep health check failed.")

    def _initialize_llm(
        self,
        allowed_node_types: Union[List[str], str],
        allowed_edge_types: Union[List[str], str],
        node_properties: Union[List[str], str],
        edge_properties: Union[List[str], str],
    ):
        """Initialize the LLM model & LLMGraphTransformer object."""

        # Process string inputs if needed
        if allowed_node_types and isinstance(allowed_node_types, str):
            allowed_node_types = allowed_node_types.split(",")

        if allowed_edge_types and isinstance(allowed_edge_types, str):
            allowed_edge_types = allowed_edge_types.split(",")

        if node_properties and isinstance(node_properties, str):
            node_properties = node_properties.split(",")

        if edge_properties and isinstance(edge_properties, str):
            edge_properties = edge_properties.split(",")

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
                allowed_nodes=allowed_node_types,
                allowed_relationships=allowed_edge_types,
                prompt=prompt_template,
                node_properties=node_properties or False,
                relationship_properties=edge_properties or False,
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
            self.embeddings = HuggingFaceEndpointEmbeddings(
                model=TEI_EMBEDDING_ENDPOINT,
                task="feature-extraction",
                huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            )
        elif TEI_EMBED_MODEL:
            self.embeddings = HuggingFaceEmbeddings(model_name=TEI_EMBED_MODEL)
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

    async def ingest_data_to_arango(
        self,
        doc_path: DocPath,
        graph_name: str,
        insert_async: bool,
        insert_batch_size: int,
        embed_nodes: bool,
        embed_edges: bool,
        embed_chunks: bool,
        include_chunks: bool,
        text_capitalization_strategy: str,
    ):
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

        content = await document_loader(path)

        structured_types = [".xlsx", ".csv", ".json", "jsonl"]
        _, ext = os.path.splitext(path)

        if ext in structured_types:
            chunks = content
        else:
            chunks = text_splitter.split_text(content)

        if doc_path.process_table and path.endswith(".pdf"):
            table_chunks = get_tables_result(path, doc_path.table_strategy)
            if table_chunks and isinstance(table_chunks, list):
                chunks = chunks + table_chunks

        if logflag:
            logger.info(f"Created {len(chunks)} chunks of the original file")

        ################################
        # Graph generation & insertion #
        ################################

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
                include_source=include_chunks,
                graph_name=graph_name,
                update_graph_definition_if_exists=False,
                batch_size=insert_batch_size,
                use_one_entity_collection=True,
                insert_async=insert_async,
                embeddings=self.embeddings,
                embedding_field="embedding",
                embed_source=embed_chunks,
                embed_nodes=embed_nodes,
                embed_relationships=embed_edges,
                capitalization_strategy=text_capitalization_strategy,
            )

            if logflag:
                logger.info(f"Chunk {i}: processed")

        if logflag:
            logger.info(f"Graph {graph_name} created with {len(chunks)} chunks.")

        return graph_name

    async def ingest_files(self, input: Union[DataprepRequest, ArangoDBDataprepRequest]):
        """Ingest files/links content into ArangoDB database.

        Save in the format of vector[768].
        Returns '{"status": 200, "message": "Data preparation succeeded"}' if successful.
        Args:
            input (DataprepRequest | ArangoDBDataprepRequest): Model containing the following parameters:
                files (Union[UploadFile, List[UploadFile]], optional): A file or a list of files to be ingested. Defaults to File(None).
                link_list (str, optional): A list of links to be ingested. Defaults to Form(None).
                chunk_size (int, optional): The size of the chunks to be split. Defaults to Form(500).
                chunk_overlap (int, optional): The overlap between chunks. Defaults to Form(100).
                process_table (bool, optional): Whether to process tables in PDFs. Defaults to Form(False).
                table_strategy (str, optional): The strategy to process tables in PDFs. Defaults to Form("fast").
                graph_name (str, optional): The name of the graph to be created. Defaults to "GRAPH".
                insert_async (bool, optional): Whether to insert data asynchronously. Defaults to False.
                insert_batch_size (int, optional): The batch size for insertion. Defaults to 1000.
                embed_nodes (bool, optional): Whether to embed nodes. Defaults to True.
                embed_edges (bool, optional): Whether to embed edges. Defaults to True.
                embed_chunks (bool, optional): Whether to embed chunks. Defaults to True.
                allowed_node_types (List[str], optional): The allowed node types. Defaults to [].
                allowed_edge_types (List[str], optional): The allowed edge types. Defaults to [].
                node_properties (List[str], optional): The node properties to be used. Defaults to ["description"].
                edge_properties (List[str], optional): The edge properties to be used. Defaults to ["description"].
                text_capitalization_strategy (str, optional): The text capitalization strategy. Defaults to "upper".
                include_chunks (bool, optional): Whether to include chunks in the graph. Defaults to True.
        """

        files = input.files
        link_list = input.link_list
        chunk_size = input.chunk_size
        chunk_overlap = input.chunk_overlap
        process_table = input.process_table
        table_strategy = input.table_strategy
        graph_name = getattr(input, "graph_name", ARANGO_GRAPH_NAME)
        insert_async = getattr(input, "insert_async", ARANGO_INSERT_ASYNC)
        insert_batch_size = getattr(input, "insert_batch_size", ARANGO_BATCH_SIZE)
        embed_nodes = getattr(input, "embed_nodes", EMBED_NODES)
        embed_edges = getattr(input, "embed_edges", EMBED_EDGES)
        embed_chunks = getattr(input, "embed_chunks", EMBED_CHUNKS)
        allowed_node_types = getattr(input, "allowed_node_types", ALLOWED_NODE_TYPES)
        allowed_edge_types = getattr(input, "allowed_edge_types", ALLOWED_EDGE_TYPES)
        node_properties = getattr(input, "node_properties", NODE_PROPERTIES)
        edge_properties = getattr(input, "edge_properties", EDGE_PROPERTIES)
        text_capitalization_strategy = getattr(input, "text_capitalization_strategy", TEXT_CAPITALIZATION_STRATEGY)
        include_chunks = getattr(input, "include_chunks", INCLUDE_CHUNKS)

        self._initialize_llm(
            allowed_node_types=allowed_node_types,
            allowed_edge_types=allowed_edge_types,
            node_properties=node_properties,
            edge_properties=edge_properties,
        )

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
                    graph_name = await self.ingest_data_to_arango(
                        DocPath(
                            path=save_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            process_table=process_table,
                            table_strategy=table_strategy,
                        ),
                        graph_name=graph_name,
                        insert_async=insert_async,
                        insert_batch_size=insert_batch_size,
                        embed_nodes=embed_nodes,
                        embed_edges=embed_edges,
                        embed_chunks=embed_chunks,
                        text_capitalization_strategy=text_capitalization_strategy,
                        include_chunks=include_chunks,
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
                    graph_name = await self.ingest_data_to_arango(
                        DocPath(
                            path=save_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            process_table=process_table,
                            table_strategy=table_strategy,
                        ),
                        graph_name=graph_name,
                        insert_async=insert_async,
                        insert_batch_size=insert_batch_size,
                        embed_nodes=embed_nodes,
                        embed_edges=embed_edges,
                        embed_chunks=embed_chunks,
                        text_capitalization_strategy=text_capitalization_strategy,
                        include_chunks=include_chunks,
                    )
                    graph_names_created.add(graph_name)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to ingest {save_path} into ArangoDB: {e}")

                if logflag:
                    logger.info(f"Successfully saved link {link}")

        result = {
            "status": 200,
            "message": f"Data preparation succeeded: {graph_names_created}",
            "graph_names": list(graph_names_created),
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

            query = """
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
        """Delete a Graph according to `file_path`.

        `file_path`:
            - A specific graph name (e.g GRAPH_1)
            - "all": delete all graphs created
        """

        if file_path == "all":
            for graph in self.db.graphs():
                self.db.delete_graph(graph["name"], drop_collections=True)
        else:
            if not self.db.has_graph(file_path):
                raise HTTPException(status_code=400, detail=f"Graph {file_path} does not exist.")

            self.db.delete_graph(file_path, drop_collections=True)

        return {"status": True}
