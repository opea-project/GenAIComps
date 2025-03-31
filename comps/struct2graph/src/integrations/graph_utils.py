# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from langchain_neo4j import Neo4jGraph
import os



class PrepareGraphDB:
    """
    A class for preparing and saving a GraphDB
    """
    def __init__(self):
         self.graph_store = self.neo4j_link()


    def neo4j_link(self):
          NEO4J_URL        = os.environ.get["NEO4J_URL"]
          NEO4J_USERNAME   = os.environ.get["NEO4J_USERNAME"]
          NEO4J_PASSWORD   = os.environ.get["NEO4J_PASSWORD"]
          NEO4J_DATABASE   = os.environ["NEO4J_DATABASE"]
          graph_store = Neo4jGraph(
              username=NEO4J_USERNAME,
              password=NEO4J_PASSWORD,
              url=NEO4J_URL,
          )
          return graph_store

    def cleanup_neo4j(self):

          # Delete everything in a database
          cypher = """
          MATCH (n) DETACH DELETE n
          """
          self.graph_store.query(cypher)

          print("## Existing graph_store schema...")
          print(self.graph_store.schema)

          print("Deleting all nodes...")
          # Match all nodes in the graph_store
          cypher = """
            MATCH (n)
            RETURN count (n)
            """
          result = self.graph_store.query(cypher)

          print("Dropping all constraints...")
          for constraint in self.graph_store.query('SHOW CONSTRAINTS'):
              self.graph_store.query(f"DROP CONSTRAINT {constraint['name']}")

          print("Dropping all indexes...")
          for index in self.graph_store.query('SHOW INDEXES'):
              print(f"Removing index {index['name']}:")
              self.graph_store.query(f"""
                  DROP INDEX `{index['name']}`
              """)

          
          print("## Blank schema...")
          self.graph_store.refresh_schema()
          print(self.graph_store.schema)
          return

    def load_graphdb(self,cypher_cmd):
         
         LOAD_FORMAT  = os.getenv('LOAD_FORMAT','CSV')
                
         if(LOAD_FORMAT == 'CSV'):
             try:
               cypher_csv_insert=cypher_cmd #os.environ.get('CYPHER_CSV_CMD')
               print(f" INSERTING CSV Cypher command : {cypher_csv_insert}")
               print("Preparing graphdb...")
               self.graph_store.query(cypher_csv_insert)
               print("GraphDB is created and saved.")
             except NameError:
               raise ValueError("Error: The variable CYPHER_CSV_CMD is not set.")
         
         if(LOAD_FORMAT == 'JSON'):
             try:
               cypher_json_insert=cypher_cmd
               print(f" INSERTING JSON Cypher command : {cypher_json_insert}")
               self.graph_store.query(cypher_json_insert)
               print(f'The following is the graph schema \n\n {self.graph_store.schema}')
               print("GraphDB is created and saved.")
             except NameError:
               raise ValueError("Error: The variable CYPHER_JSON_CMD is not set.")
         
         print("Preparing graphdb...")
         return self.graph_store

    def prepare_insert_graphdb(self,cypher_cmd):
         print(f'Cleaning up graph db')
         self.cleanup_neo4j()
         print(f'Done cleaning up graph db')
         self.load_graphdb(cypher_cmd)
         print(f'Completed inserting into graphdb')
         print(f'The following is the graph schema \n\n {self.graph_store.schema}')
         print("Preparing graphdb...")
         print("GraphDB is created and saved.")
         return self.graph_store
