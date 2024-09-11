# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os

from langchain_community.graphs import FalkorDBGraph

falkordb_host = os.getenv("FALKORDB_HOST", "localhost")
falkordb_port = int(os.getenv("FALKORDB_PORT", "6379"))
falkordb_database = int(os.getenv("FALKORDB_DATABASE", "falkordb"))
falkordb_username = os.getenv("FALKORDB_USERNAME", "")
falkordb_password = os.getenv("FALKORDB_PASSWORD", "")
graph = FalkorDBGraph(falkordb_database, falkordb_host, falkordb_port, falkordb_username, falkordb_password)

# remove all nodes
graph.query("MATCH (n) DETACH DELETE n")

# ingest
import_query = json.load(open("../data/microservices.json", "r"))["query"]
graph.query(import_query)
print("Total nodes: ", graph.query("MATCH (n) RETURN count(n)"))
print("Total edges: ", graph.query("MATCH ()-->() RETURN count(*)"))
