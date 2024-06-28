import motor.motor_asyncio as motor
from config import MONGO_HOST, MONGO_PORT, DB_NAME
from typing import Any

class MongoClient:
    conn_url = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/"
    @staticmethod
    def get_db_client() -> Any:
        try:
            client = motor.AsyncIOMotorClient(MongoClient.conn_url)
            db = client[DB_NAME]
            return db

        except Exception as e:
            print(e)
            raise Exception()