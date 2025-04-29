# Dataprep Microservice

The Dataprep Microservice aims to preprocess the data from various sources (either structured or unstructured data) to text data, and convert the text data to embedding vectors then store them in the database.

## Install Requirements

```bash
apt-get update
apt-get install libreoffice
```

## Use LVM (Large Vision Model) for Summarizing Image Data

Occasionally unstructured data will contain image data, to convert the image data to the text data, LVM can be used to summarize the image. To leverage LVM, please refer to this [readme](../lvms/src/README.md) to start the LVM microservice first and then set the below environment variable, before starting any dataprep microservice.

```bash
export SUMMARIZE_IMAGE_VIA_LVM=1
```

## Dataprep Microservice with Redis

For details, please refer to this [readme](src/README_redis.md)

## Dataprep Microservice with Milvus

For details, please refer to this [readme](src/README_milvus.md)

## Dataprep Microservice with Qdrant

For details, please refer to this [readme](src/README_qdrant.md)

## Dataprep Microservice with Pinecone

For details, please refer to this [readme](src/README_pinecone.md)

## Dataprep Microservice with PGVector

For details, please refer to this [readme](src/README_pgvector.md)

## Dataprep Microservice with VDMS

For details, please refer to this [readme](src/README_vdms.md)

## Dataprep Microservice with Multimodal

For details, please refer to this [readme](src/README_multimodal.md)

## Dataprep Microservice with ElasticSearch

For details, please refer to this [readme](src/README_elasticsearch.md)

## Dataprep Microservice with OpenSearch

For details, please refer to this [readme](src/README_opensearch.md)

## Dataprep Microservice with neo4j

For details, please refer to this [readme](src/README_neo4j_llamaindex.md)

## Dataprep Microservice for financial domain data

For details, please refer to this [readme](src/README_finance.md)

## Dataprep Microservice with MariaDB Vector

For details, please refer to this [readme](src/README_mariadb.md)