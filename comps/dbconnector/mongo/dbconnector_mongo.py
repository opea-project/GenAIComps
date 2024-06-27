# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from comps.cores.mega.micro_service import opea_microservices, register_microservice
from comps.cores.proto.api_protocol import ChatCompletionRequest
from mongo_store import DocumentStore
from pydantic import BaseModel
from uuid import uuid4


class ChatMessage(BaseModel):
    user: str
    data: ChatCompletionRequest
    first_query : Optional[str] = None
    id: Optional[str] = None

class ChatId(BaseModel):
    user: str
    id: Optional[str] = None
    
    
    
def get_first_string(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, list):
        # Assuming we want the first string from the first dictionary
        if value and isinstance(value[0], dict):
            first_dict = value[0]
            if first_dict:
                # Get the first value from the dictionary
                first_key = next(iter(first_dict))
                return first_dict[first_key]
            
@register_microservice(
    name="opea_service@dbconnector_mongo_create",
    endpoint="/v1/dbconnector/create",
    host="0.0.0.0",
    input_datatype=ChatMessage,
    port=6012
)
async def create_documents(document: ChatMessage):
    """Process the mongo document."""
    store = DocumentStore(document.user)
    store.initialize_storage()
    if document.first_query is None:
        document.first_query = get_first_string(document.data.messages)
    if document.id is None:
        document.id = uuid4()
        res = await store.save_document(document)
    else:
        res = await store.update_document(document.id, document.data, document.first_query)
    return res
    
@register_microservice(
    name="opea_service@dbconnector_mongo_get",
    endpoint="/v1/dbconnector/get",
    host="0.0.0.0",
    input_datatype=ChatId,
    port=6013,
)
async def get_documents(document: ChatId):
    """Process the mongo document."""
    store = DocumentStore(document.user)
    store.initialize_storage()
    if document.id is None:
        res = await store.get_all_documents_of_user()
        
    else:
        res = await store.get_user_documents_by_id(document.id)
    return res

if __name__ == "__main__":
    opea_microservices["opea_service@dbconnector_mongo_get"].start()
    opea_microservices["opea_service@dbconnector_mongo_create"].start()
