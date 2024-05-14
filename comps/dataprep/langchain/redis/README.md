# Start Microservices

## 1. Start Redis Stack server

Please refer to this [readme](../../../vectorstores/langchain/redis/README.md).

## 2. Start document preparation microservice for Redis
Start document preparation microservice for Redis with below command.
```bash
python prepare_doc_redis.py
```

# Invoke Microservices

Once document preparation microservice for Redis is started, user can use below command to invoke the microservice to convert the document to embedding and save to the database.
```bash
curl -X POST -H "Content-Type: application/json" -d '{"path":"/path/to/document"}' http://localhost:6000/v1/dataprep
```