# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pathlib
import sys
cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../../../")
sys.path.append(comps_path)
from langchain_community.graphs import Neo4jGraph

from langsmith import traceable
from comps import GraphDoc, TextDoc, opea_microservices, register_microservice, ServiceType



@register_microservice(
    name="opea_service@knowledge_graph",
    endpoint="/v1/graphs",
    host="0.0.0.0",
    port=8060,
)
def graph_query(input: GraphDoc) -> TextDoc:
    neo4j_endpoint = os.getenv("NEO4J_ENDPOINT", "neo4j://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")

    graph = Neo4jGraph(
        url=neo4j_endpoint,
        username=neo4j_username,
        password=neo4j_password
    )

    if input.strtype == "query":
        return TextDoc(text="Query with LLM will be implemented later")
    elif input.strtype == "cypher":
        result = graph.query(input.text)
        return TextDoc(text=result)
    elif input.strtype == "rag":
        from langchain_community.vectorstores.neo4j_vector import Neo4jVector
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.llms import HuggingFaceEndpoint
        from langchain.chains import RetrievalQA

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        index_name = None
        node_label = None
        text_node_properties = None
        embedding_node_property = None
        if input.ragparam is not None:
            index_name = input.ragparam.index_name
            node_label = input.ragparam.node_label
            text_node_properties = input.ragparam.text_node_properties
            embedding_node_property = input.ragparam.embedding_node_property
        vector_index = Neo4jVector.from_existing_graph(
            embeddings,
            url=neo4j_endpoint,
            username=neo4j_username,
            password=neo4j_password,
            index_name=index_name,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property=embedding_node_property,
        )

        llm_endpoint = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
        llm = HuggingFaceEndpoint(
            endpoint_url=llm_endpoint,
            timeout=600,
        )
        
        vector_qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vector_index.as_retriever())
        
        result = vector_qa.invoke(input.text)
    return TextDoc(text=result)

if __name__ == "__main__":
    opea_microservices["opea_service@knowledge_graph"].start()