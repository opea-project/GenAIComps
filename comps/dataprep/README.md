# Dataprep Microservice

The Dataprep Microservice aims to preprocess the data from various sources (either structured or unstructured data) to text data, and convert the text data to embedding vectors then store them in the database.

## Table of contents

1. [Install Requirements](#install-requirements)
2. [Summarizing Image Data with LVM](#summarizing-image-data-with-lvm)
3. [Dataprep Microservice on Various Databases](#dataprep-microservice-on-various-databases)
4. [Running in the air gapped environment](#running-in-the-air-gapped-environment)

## Install Requirements

```bash
apt-get update
apt-get install libreoffice
```

## Summarizing Image Data with LVM

Occasionally unstructured data will contain image data, to convert the image data to the text data, LVM (Large Vision Model) can be used to summarize the image. To leverage LVM, please refer to this [readme](../lvms/README.md) to start the LVM microservice first and then set the below environment variable, before starting any dataprep microservice.

```bash
export SUMMARIZE_IMAGE_VIA_LVM=1
```

## Dataprep Microservice on Various Databases

Dataprep microservice are supported on various databases, as shown in the table below, for details, please refer to the respective readme listed below.

| Databases               | Readme                                                                   |
| :---------------------- | :----------------------------------------------------------------------- |
| `Redis`                 | [Dataprep Microservice with Redis](src/README_redis.md)                  |
| `Milvus`                | [Dataprep Microservice with Milvus](src/README_milvus.md)                |
| `Qdrant`                | [Dataprep Microservice with Qdrant](src/README_qdrant.md)                |
| `Pinecone`              | [Dataprep Microservice with Pinecone](src/README_pinecone.md)            |
| `PGVector`              | [Dataprep Microservice with PGVector](src/README_pgvector.md)            |
| `VDMS`                  | [Dataprep Microservice with VDMS](src/README_vdms.md)                    |
| `Multimodal`            | [Dataprep Microservice with Multimodal](src/README_multimodal.md)        |
| `ElasticSearch`         | [Dataprep Microservice with ElasticSearch](src/README_elasticsearch.md)  |
| `OpenSearch`            | [Dataprep Microservice with OpenSearch](src/README_opensearch.md)        |
| `neo4j`                 | [Dataprep Microservice with neo4j](src/README_neo4j_llamaindex.md)       |
| `financial domain data` | [Dataprep Microservice for financial domain data](src/README_finance.md) |
| `MariaDB`               | [Dataprep Microservice with MariaDB Vector](src/README_mariadb.md)       |
| `ArangoDB`              | [Dataprep Microservice with ArangoDB Vector](src/README_arangodb.md)     |

## Running in the air gapped environment

The following steps are common for running the dataprep microservice in an air gapped environment (a.k.a. environment with no internet access), for all DB backends.

1. Download the following models, e.g. `huggingface-cli download --cache-dir <model data directory> <model>`

- microsoft/table-transformer-structure-recognition
- timm/resnet18.a1_in1k
- unstructuredio/yolo_x_layout

2. launch the `dataprep` microservice with the following settings:

- mount the `model data directory` as the `/data` directory within the `dataprep` container
- set environment variable `HF_HUB_OFFLINE` to 1 when launching the `dataprep` microservice

e.g. `docker run -d -v <model data directory>:/data -e HF_HUB_OFFLINE=1 ... ...`
