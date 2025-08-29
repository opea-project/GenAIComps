# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from comps.cores.proto.api_protocol import ChatCompletionRequest
from comps.cores.storages.models import ChatId, ChatMessage
from comps.cores.storages.stores import get_store


class ChatMessageDto(BaseModel):
    data: ChatCompletionRequest
    first_query: Optional[str] = None
    doc_id: Optional[str] = None
    user: Optional[str] = None


def _preprocess(document: ChatMessage) -> dict:
    """Converts a ChatMessage object to a dictionary suitable for storage.

    Args:
        document (ChatMessage): The ChatMessage object to be converted.

    Returns:
        dict: A dictionary representation of the ChatMessage, ready for storage.
    """
    return {
        "data": document.data.model_dump(by_alias=True, mode="json"),
        "first_query": document.first_query,
        "doc_id": document.id,
        "user": document.data.user,
    }


def _postprocess(rs: dict) -> dict:
    return rs.get("data")


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
        return await store.aupdate_document(_preprocess(document))
    else:
        return await store.asave_document(_preprocess(document))


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
        return await store.aget_documents_by_user(document.user)
    else:
        rs = await store.aget_document_by_id(document.id)
        return _postprocess(rs)


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
