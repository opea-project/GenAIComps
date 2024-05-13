#!/bin/sh

cd /home/user/comps/retrievers/langchain
python ingest.py

python retriever_redis.py
