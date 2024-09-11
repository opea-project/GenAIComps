# Knowledge Graph Microservice

This microservice, designed for efficiently handling and retrieving informantion from knowledge graph. The microservice integrates text retriever, knowledge graph quick search and LLM agent, which can be combined to enhance question answering.

The service contains three modes:

- "cypher": Query knowledge graph directly with cypher
- "rag": Apply similarity search on embeddings of knowledge graph
- "query": An LLM agent will automatically choose tools (RAG or CypherChain) to enhance the question answering

# Knowledge Graph Microservice with Neo4J

For details, please refer to this [readme](langchain/neo4j/README.md)

# Knowledge Graph Microservice with FalkorDB

For details, please refer to this [readme](langchain/falkordb/README.md)
