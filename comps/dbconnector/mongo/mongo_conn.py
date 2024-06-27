import motor.motor_asyncio as motor
from config import MONGO_HOST, MONGO_PORT, DB_NAME
from typing import Any

class MongoClient:
    mongo_host = MONGO_HOST
    mongo_port = MONGO_PORT
    conn_url = f"mongodb://{mongo_host}:{mongo_port}/"
    db_name = DB_NAME

    @staticmethod
    def get_db_client() -> Any:
        try:
            client = motor.AsyncIOMotorClient(MongoClient.conn_url)
            db = client[DB_NAME]
            return db

        except Exception as e:
            print(e)
            raise Exception()