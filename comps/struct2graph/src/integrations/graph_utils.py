# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from langchain_neo4j import Neo4jGraph

from comps import CustomLogger

logger = CustomLogger("opea_struct2graph")


class PrepareGraphDB:
    """A class for preparing and saving a GraphDB."""

    def __init__(self):
        self.graph_store = self.neo4j_link()

    def neo4j_link(self):
        NEO4J_URL = os.getenv("NEO4J_URL")
        NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

        if not all([NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD]):
            raise EnvironmentError("Missing required Neo4j environment variables")

        graph_store = Neo4jGraph(username=NEO4J_USERNAME, password=NEO4J_PASSWORD, url=NEO4J_URL)
        return graph_store

    def cleanup_neo4j(self):
        try:
            cypher = """MATCH (n) DETACH DELETE n"""
            self.graph_store.query(cypher)

            logger.info("## Existing graph_store schema...")
            logger.info(self.graph_store.schema)

            logger.info("Deleting all nodes...")
            cypher = """MATCH (n) RETURN count(n)"""
            result = self.graph_store.query(cypher)

            logger.info("Dropping all constraints...")
            for constraint in self.graph_store.query("SHOW CONSTRAINTS"):
                self.graph_store.query(f"DROP CONSTRAINT {constraint['name']}")

            logger.info("Dropping all indexes...")
            for index in self.graph_store.query("SHOW INDEXES"):
                logger.info(f"Removing index {index['name']}:")
                self.graph_store.query(f"""DROP INDEX `{index['name']}`""")

            logger.info("## Blank schema...")
            self.graph_store.refresh_schema()
            logger.info(self.graph_store.schema)
            return

        except Exception as e:
            logger.error(f"Failed to cleanup Neo4j database: {str(e)}")
            raise

    def load_graphdb(self, cypher_cmd):
        LOAD_FORMAT = os.getenv("LOAD_FORMAT", "CSV")

        try:
            if LOAD_FORMAT == "CSV":
                cypher_csv_insert = cypher_cmd
                logger.info(f"INSERTING CSV Cypher command : {cypher_csv_insert}")
                logger.info("Preparing graphdb...")
                self.graph_store.query(cypher_csv_insert)
                logger.info("GraphDB is created and saved.")

            elif LOAD_FORMAT == "JSON":
                cypher_json_insert = cypher_cmd
                logger.info(f"INSERTING JSON Cypher command : {cypher_json_insert}")
                self.graph_store.query(cypher_json_insert)
                logger.info(f"The following is the graph schema \n\n {self.graph_store.schema}")
                logger.info("GraphDB is created and saved.")

            else:
                logger.error("Only CSV and JSON formats are supported")
                raise ValueError("Only CSV and JSON formats are supported")

            logger.info("Preparing graphdb...")
            return self.graph_store

        except NameError:
            raise ValueError("Error: The variable CYPHER_CSV_CMD is not set.")

    def prepare_insert_graphdb(self, cypher_cmd):
        logger.info("Cleaning up graph db")
        self.cleanup_neo4j()
        logger.info("Done cleaning up graph db")
        self.load_graphdb(cypher_cmd)
        logger.info("Completed inserting into graphdb")
        logger.info(f"The following is the graph schema \n\n {self.graph_store.schema}")
        logger.info("Preparing graphdb...")
        logger.info("GraphDB is created and saved.")
        return self.graph_store
