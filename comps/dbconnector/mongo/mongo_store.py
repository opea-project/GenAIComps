from bson.objectid import ObjectId
import bson.errors as BsonError
from bson import json_util
import json
from config import MONGO_HOST, MONGO_PORT, COLLECTION_NAME, DB_NAME
import motor.motor_asyncio as motor
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

class DocumentStore:

    def __init__(
        self,
        user: str,
    ):
        self.user = user

    def initialize_storage(self) -> None:
        self.db_client = MongoClient.get_db_client()
        self.collection = self.db_client[COLLECTION_NAME]

    async def save_document(self, document):
        """Stores a new document into the storage."""

        try:
            inserted_conv = await self.collection.insert_one(
                document.model_dump(by_alias=True, mode="json", exclude={"document_id"})
            )
            document_id = str(inserted_conv.inserted_id)
            return document_id

        except Exception as e:
            print(e)
            raise Exception(e)
        
    async def update_document(self, document_id, updated_data, first_query) -> bool:
        try:
            _id = ObjectId(document_id)
            update_result = await self.collection.update_one(
                {"_id": _id, "user": self.user},
                {"$set": {"data": updated_data.model_dump(by_alias=True, mode="json"), "first_query": first_query}}
            )
            if update_result.modified_count == 1:
                return True
            else:
                return False
        except BsonError.InvalidId as e:
            print(e)
            raise KeyError(e)
        except Exception as e:
            print(e)
            raise Exception(e)

    async def get_all_documents_of_user(self) -> list[dict]:
        conversation_list: list = []
        try:
            cursor = self.collection.find({"user": self.user}, {"data": 0})
            async for document in cursor:
                conversation_list.append(document)
            response = json.loads(json_util.dumps(conversation_list))
            return response

        except Exception as e:
            print(e)
            raise Exception(e)

    async def get_user_documents_by_id(self, document_id) -> dict | None:
        try:
            _id = ObjectId(document_id)
            print("ID: ", _id)
            response: dict | None = await self.collection.find_one({"_id": _id, "user": self.user})
            #TODO this is a hack, need to fix this
            if response:
                response = json.loads(json_util.dumps(response))
                return response["data"]
            return None

        except BsonError.InvalidId as e:
            print(e)
            raise KeyError(e)

        except Exception as e:
            print(e)
            raise Exception(e)

    async def delete_document(self, document_id) -> bool:

        try:
            _id = ObjectId(document_id)
            delete_result = await self.collection.delete_one({"_id": _id, "user_id": self.user})

            delete_count = delete_result.deleted_count
            print(f"Deleted {delete_count} documents!")

            return True if delete_count == 1 else False

        except BsonError.InvalidId as e:
            print(e)
            raise KeyError(e)

        except Exception as e:
            print(e)
            raise Exception(e)
