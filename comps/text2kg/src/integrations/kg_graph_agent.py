# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
from typing import Literal

import neo4j
import nest_asyncio
from llama_index.core import KnowledgeGraphIndex, Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore, Neo4jPropertyGraphStore
from llama_index.llms.huggingface import HuggingFaceLLM

nest_asyncio.apply()


class GenerateKG:
    def __init__(self, llm, embedding_model, data_directory):
        self.data_directory = data_directory
        self.llm = llm
        self.embed_model = embedding_model
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        self.NEO4J_URL = os.environ.get["NEO4J_URL"]
        self.NEO4J_URI = os.environ.get["NEO4J_URI"]
        self.NEO4J_USERNAME = os.environ.get["NEO4J_USERNAME"]
        self.NEO4J_PASSWORD = os.environ.get["NEO4J_PASSWORD"]
        self.NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
        print(" loading and preparing llm and embedding models")

    def __load_docs(self):

        DATA_DIRECTORY = os.environ.get["DATA_DIRECTORY"]
        reader = SimpleDirectoryReader(input_dir=DATA_DIRECTORY)
        documents = reader.load_data()
        print("loading documents")

        return documents

    # -------------------------------------------------------------------------------
    #   Link up to Neo4j
    # -------------------------------------------------------------------------------
    def __neo4j_link(self):

        graph_store = Neo4jGraphStore(
            username=self.NEO4J_USERNAME,
            password=self.NEO4J_PASSWORD,
            url=self.NEO4J_URL,
            database=self.NEO4J_DATABASE,
        )
        return graph_store

    def __graph_index(self, documents, llm, embed_model, graph_store):

        entities = os.environ.get("ENTITIES", "").split(",")
        relations = os.environ.get("RELATIONS", "").split(",")

        # Get and parse validation schema
        validation_schema_str = os.environ.get("VALIDATION_SCHEMA", "{}")
        try:
            validation_schema = json.loads(validation_schema_str)
        except json.JSONDecodeError:
            print("Warning: Could not parse VALIDATION_SCHEMA")
            validation_schema = {}

        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        neo4j_index = KnowledgeGraphIndex.from_documents(
            documents=documents,
            max_triplets_per_chunk=3,
            storage_context=storage_context,
            embed_model=embed_model,
            include_embeddings=True,
        )
        return neo4j_index

    def __create_index(self, documents, embed_model, llm):
        """Creates index in neo4j database."""
        graph_store = self.__neo4j_link(
            self.NEO4J_URL, self.NEO4J_URI, self.NEO4J_USERNAME, self.NEO4J_PASSWORD, self.NEO4J_DATABASE
        )
        neo4j_index = self.__graph_index(documents, llm, embed_model, graph_store)
        print(f" neo4j index {neo4j_index.index_struct}")
        print("creating graph index for documents")
        return neo4j_index

    def prepare_and_save_graphdb(self):
        """Load, chunk, and create graph and load it into neo4j database."""
        print("entering prepare and save for structured data")
        docs = self.__load_docs()
        neo4j_index = self.__create_index(docs, self.embed_model, self.llm)
        print("Preparing graphdb...")
        print("GraphDB is created and saved.")
        return neo4j_index
