# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import requests
from langchain_community.graphs import Neo4jGraph

url = "neo4j://localhost:7687"
username = "neo4j"
password = "neo4j"
graph = Neo4jGraph(url=url, username=username, password=password)

import_url = "https://gist.githubusercontent.com/tomasonjo/08dc8ba0e19d592c4c3cde40dd6abcc3/raw/e90b0c9386bf8be15b199e8ac8f83fc265a2ac57/microservices.json"
import_query = requests.get(import_url).json()["query"]
graph.query(import_query)
