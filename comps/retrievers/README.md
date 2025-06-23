# Retriever Microservice

This retriever microservice is a highly efficient search service designed for handling and retrieving embedding vectors. It operates by receiving an embedding vector as input and conducting a similarity search against vectors stored in a VectorDB database. Users must specify the VectorDB's URL and the index name, and the service searches within that index to find documents with the highest similarity to the input vector.

The service primarily utilizes similarity measures in vector space to rapidly retrieve contextually similar documents. The vector-based retrieval approach is particularly suited for handling large datasets, offering fast and accurate search results that significantly enhance the efficiency and quality of information retrieval.

Overall, this microservice provides robust backend support for applications requiring efficient similarity searches, playing a vital role in scenarios such as recommendation systems, information retrieval, or any other context where precise measurement of document similarity is crucial.

## Retriever Microservice with Redis

For details, please refer to this [readme](src/README_redis.md)

## Retriever Microservice with Milvus

For details, please refer to this [readme](src/README_milvus.md)

## Retriever Microservice with Qdrant

For details, please refer to this [readme](src/README_qdrant.md)

## Retriever Microservice with PGVector

For details, please refer to this [readme](src/README_pgvector.md)

## Retriever Microservice with VDMS

For details, please refer to this [readme](src/README_vdms.md)

## Retriever Microservice with ElasticSearch

For details, please refer to this [readme](src/README_elasticsearch.md)

## Retriever Microservice with OpenSearch

For details, please refer to this [readme](src/README_opensearch.md)

## Retriever Microservice with neo4j

For details, please refer to this [readme](src/README_neo4j.md)

## Retriever Microservice with Pathway

For details, please refer to this [readme](src/README_pathway.md)

## Retriever Microservice with MariaDB Vector

For details, please refer to this [readme](src/README_mariadb.md)

## Retriever Microservice with ArangoDB

For details, please refer to this [readme](src/README_arangodb.md)
