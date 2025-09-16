# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os

from fastapi import HTTPException

from comps import CustomLogger
from comps.chathistory.src.integrations.data_store import delete, get, save_or_update
from comps.cores.mega.constants import MCPFuncType
from comps.cores.mega.micro_service import opea_microservices, register_microservice
from comps.cores.storages.models import ChatId, ChatMessage
from comps.cores.storages.stores import get_store_name

logger = CustomLogger(f"chathistory_{get_store_name()}")
logflag = os.getenv("LOGFLAG", False)
enable_mcp = os.getenv("ENABLE_MCP", "").strip().lower() in {"true", "1", "yes"}


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
    name=f"opea_service@chathistory_{get_store_name()}",
    endpoint="/v1/chathistory/create",
    host="0.0.0.0",
    input_datatype=ChatMessage,
    port=6012,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Create or update chat conversation history for AI agent workflows",
)
async def create_documents(document: ChatMessage):
    """Creates or updates a document in the document store.

    Args:
        document (ChatMessage): The ChatMessage object containing the data to be stored.

    Returns:
        The result of the operation if successful, None otherwise.
    """
    if logflag:
        logger.info(document)
    try:
        if document.first_query is None:
            document.first_query = get_first_string(document.data.messages)
        res = await save_or_update(document)
        if logflag:
            logger.info(res)
        return res
    except Exception as e:
        # Handle the exception here
        logger.info(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@register_microservice(
    name=f"opea_service@chathistory_{get_store_name()}",
    endpoint="/v1/chathistory/get",
    host="0.0.0.0",
    input_datatype=ChatId,
    port=6012,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Retrieve chat conversation history for AI agent workflows",
)
async def get_documents(document: ChatId):
    """Retrieves documents from the document store based on the provided ChatId.

    Args:
        document (ChatId): The ChatId object containing the user and optional document id.

    Returns:
        The retrieved documents if successful, None otherwise.
    """
    if logflag:
        logger.info(document)
    try:
        res = await get(document)
        if logflag:
            logger.info(res)
        return res
    except Exception as e:
        # Handle the exception here
        logger.info(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@register_microservice(
    name=f"opea_service@chathistory_{get_store_name()}",
    endpoint="/v1/chathistory/delete",
    host="0.0.0.0",
    input_datatype=ChatId,
    port=6012,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Delete chat conversation history for AI agent workflows",
)
async def delete_documents(document: ChatId):
    """Deletes a document from the document store based on the provided ChatId.

    Args:
        document (ChatId): The ChatId object containing the user and document id.

    Returns:
        The result of the deletion if successful, None otherwise.
    """
    if logflag:
        logger.info(document)
    try:
        res = await delete(document)
        if logflag:
            logger.info(res)
        return res
    except Exception as e:
        # Handle the exception here
        logger.info(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    opea_microservices[f"opea_service@chathistory_{get_store_name()}"].start()
