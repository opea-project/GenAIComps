# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from comps.cores.proto.api_protocol import ChatCompletionRequest
from comps.cores.storages.models import ChatId, ChatMessage
from comps.cores.storages.stores import get_store, postget, prepersist


class ChatMessageDto(BaseModel):
    data: ChatCompletionRequest
    first_query: Optional[str] = None
    doc_id: Optional[str] = None
    user: Optional[str] = None


def _prepersist(document: ChatMessage) -> dict:
    """Converts a ChatMessage object to a dictionary suitable for persistence.

    Args:
        document (ChatMessage): The ChatMessage object to be converted.

    Returns:
        dict: A dictionary representation of the ChatMessage, ready for persistence.
    """
    data_dict = document.model_dump(by_alias=True, mode="json")
    data_dict = prepersist("id", data_dict)
    return data_dict


def _post_getby_id(rs: dict) -> dict:
    """Post-processes a document retrieved by ID from the store.

    Args:
        rs (dict): The raw document dictionary from the store.

    Returns:
        dict: The processed document data, or None if the document doesn't exist.
    """
    rs = postget("id", rs)
    return rs.get("data") if rs else None


def _post_getby_user(rss: list) -> list:
    """Post-processes a list of documents retrieved by user from the store.

    Args:
        rss (list): A list of raw document dictionaries from the store.

    Returns:
        list: A list of processed documents with the 'data' field removed.
    """
    for rs in rss:
        rs = postget("id", rs)
        rs.pop("data")
    return rss


def _check_user_info(document: ChatMessage | ChatId):
    """Checks if the user information is provided in the document.

    Args:
        document (ChatMessage|ChatId): The document to be checked.
    Raises:
        HTTPException: If the user information is missing.
    """
    user = document.data.user if isinstance(document, ChatMessage) else document.user
    if user is None:
        raise HTTPException(status_code=400, detail="Please provide the user information")


async def save_or_update(document: ChatMessage):
    """Saves a new chat message or updates an existing one in the data store.

    Args:
        document (ChatMessage): The ChatMessage object to be saved or updated.
                               If the document has an ID, it will be updated;
                               otherwise, a new document will be created.

    Returns:
        The result of the save or update operation from the store.
    """
    _check_user_info(document)
    store = get_store(document.data.user)
    if document.id:
        return await store.aupdate_document(_prepersist(document))
    else:
        return await store.asave_document(_prepersist(document))


async def get(document: ChatId):
    """Retrieves chat messages from the data store.

    Args:
        document (ChatId): The ChatId object containing user information and
                           optionally a document ID. If document.id is None,
                           retrieves all documents for the user; otherwise,
                           retrieves the specific document by ID.

    Returns:
        Either a list of all documents for the user (if document.id is None) or
        a specific document (if document.id is provided).
    """
    _check_user_info(document)
    store = get_store(document.user)
    if document.id is None:
        rss = await store.asearch(key="data.user", value=document.user)
        return _post_getby_user(rss)
    else:
        rs = await store.aget_document_by_id(document.id)
        return _post_getby_id(rs)


async def delete(document: ChatId):
    """Deletes a specific chat message from the data store.

    Args:
        document (ChatId): The ChatId object containing user information and document ID.
                          The document ID must be provided for deletion.

    Returns:
        The result of the delete operation from the store.

    Raises:
        Exception: If the document ID is not provided.
    """
    _check_user_info(document)
    store = get_store(document.user)
    if document.id is None:
        raise Exception("Document id is required.")
    else:
        return await store.adelete_document(document.id)
