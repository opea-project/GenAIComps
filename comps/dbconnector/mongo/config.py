import os

# MONGO configuration
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PORT", 27017)
DB_NAME = os.getenv("DB_NAME", "chat")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chat")
