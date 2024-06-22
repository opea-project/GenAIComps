# DBConnector Microservice

DB Connector microservice helps us to connenct with Database and save the user conversation. 

## Setup Environment Variables

```bash
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export MONGO_HOST=${MONGO_HOST}
export MONGO_HOST=27017
export DB_NAME=${DB_NAME}
export COLLECTION_NAME=${COLLECTION_NAME}
```

## Start DBConnector Microservice for MongoDB with Python Script

Start document preparation microservice for Milvus with below command.

```bash
python dbconnector_mongo.py
```

# 🚀Start Microservice with Docker

## Build Docker Image

```bash
cd ../../../../
docker build -t opea/dbconnector-mongo-server:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dbconnector/mongo/docker/Dockerfile .
```

## Run Docker with CLI

Run mongoDB image

```bash
docker run -d -p 27017:27017 --name=mongo mongo:latest
```
Run the DBConnector Service

```bash
docker run --name="dbconnector-mongo-server" -p 6013:6013 -p 6012:6012 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e MONGO_HOST=${MONGO_HOST} -e MONGO_PORT=${MONGO_PORT} -e DB_NAME=${DB_NAME} -e COLLECTION_NAME=${COLLECTION_NAME} opea/dbconnector-mongo-server:latest
```

# Invoke Microservice

Once document preparation microservice for Qdrant is started, user can use below command to invoke the microservice to convert the document to embedding and save to the database.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"path":"/home/user/doc/your_document_name"}' http://localhost:6010/v1/dataprep
```

You can specify chunk_size and chunk_size by the following commands.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"path":"/home/user/doc/your_document_name","chunk_size":1500,"chunk_overlap":100}' http://localhost:6010/v1/dataprep
```
