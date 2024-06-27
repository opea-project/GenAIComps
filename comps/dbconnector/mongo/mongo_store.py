from bson.objectid import ObjectId
import bson.errors as BsonError
from bson import json_util
import json
from config import COLLECTION_NAME
from mongo_conn import MongoClient

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
                document.model_dump(by_alias=True, mode="json", exclude={"id"})
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
                document_id = str(update_result.inserted_id)
                return document_id
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
                document["id"] = str(document["_id"])
                del document["_id"]
                conversation_list.append(document)
            return conversation_list

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
                del response["_id"]
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
