# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from comps.cores.proto.api_protocol import ChatCompletionRequest
from comps.cores.storages.models import ChatFeedback, FeedbackData, FeedbackId
from comps.cores.storages.stores import get_store, postget, prepersist


class ChatFeedbackDto(BaseModel):
    chat_data: ChatCompletionRequest
    feedback_data: FeedbackData
    chat_id: Optional[str] = None
    feedback_id: Optional[str] = None
    user: str


def _prepersist(feedback: ChatFeedback) -> dict:
    """Converts a ChatFeedback object to a dictionary suitable for persistence.

    Args:
        feedback (ChatFeedback): The ChatFeedback object to be converted.

    Returns:
        dict: A dictionary representation of the ChatFeedback, ready for persistence.
    """
    data_dict = feedback.model_dump(by_alias=True, mode="json")
    data_dict = prepersist("feedback_id", data_dict)
    return data_dict


def _post_getby_id(rs: dict) -> dict:
    """Processes a single feedback record after retrieval by ID.

    Converts the internal 'id' field back to 'feedback_id' for external use.

    Args:
        rs (dict): The raw feedback record dictionary from the store.

    Returns:
        dict: The processed feedback record with proper field naming.
    """
    rs = postget("feedback_id", rs)
    return rs


def _post_getby_user(rss: list) -> list:
    """Processes multiple feedback records after retrieval by user.

    Converts the internal 'id' field back to 'feedback_id' for each record in the list.

    Args:
        rss (list): A list of raw feedback record dictionaries from the store.

    Returns:
        list: The list of processed feedback records with proper field naming.
    """
    for rs in rss:
        rs = postget("feedback_id", rs)
    return rss


def _check_user_info(feedback: ChatFeedback | FeedbackId):
    """Checks if the user information is provided in the document.

    Args:
        feedback (ChatFeedback|FeedbackId): The feedback to be checked.

    Raises:
        HTTPException: If the user information is missing.
    """
    user = feedback.chat_data.user if isinstance(feedback, ChatFeedback) else feedback.user
    if user is None or (isinstance(user, str) and user.strip() == ""):
        raise HTTPException(status_code=400, detail="User information is required but not provided")


async def save_or_update(feedback: ChatFeedback):
    """Saves a new feedback record or updates an existing one.

    This function determines whether to create a new feedback record or update
    an existing one based on the presence of feedback_id. If feedback_id is None,
    a new record is created; otherwise, the existing record is updated.

    Args:
        feedback (ChatFeedback): The ChatFeedback object to be saved or updated.

    Returns:
        The result from the store operation (save or update).

    Raises:
        HTTPException: If user information is missing.
    """
    _check_user_info(feedback)
    store = get_store(feedback.chat_data.user)
    if feedback.feedback_id is None:
        return await store.asave_document(_prepersist(feedback))
    else:
        return await store.aupdate_document(_prepersist(feedback))


async def get(feedback: FeedbackId):
    """Retrieves feedback record(s) based on the provided FeedbackId.

    This function can retrieve either a specific feedback record by its ID
    or all feedback records for a user. If feedback_id is provided, it returns
    the specific record; otherwise, it returns all records for the user.

    Args:
        feedback (FeedbackId): The FeedbackId object containing user and optional feedback_id.

    Returns:
        Either a specific feedback document (if feedback_id provided) or a list of
        all feedback documents for the user.

    Raises:
        HTTPException: If user information is missing.
    """
    _check_user_info(feedback)
    store = get_store(feedback.user)
    if feedback.feedback_id:
        rs = await store.aget_document_by_id(feedback.feedback_id)
        return _post_getby_id(rs)
    else:
        rss = await store.asearch(key="chat_data.user", value=feedback.user)
        return _post_getby_user(rss)


async def delete(feedback: FeedbackId):
    """Deletes a specific feedback record from the store.

    This function removes a feedback record identified by the feedback_id.
    The feedback_id must be provided and cannot be None.

    Args:
        feedback (FeedbackId): The FeedbackId object containing user and feedback_id.

    Returns:
        The result from the store delete operation.

    Raises:
        HTTPException: If user information is missing.
        Exception: If feedback_id is None or not provided.
    """
    _check_user_info(feedback)
    store = get_store(feedback.user)
    if feedback.feedback_id is None:
        raise Exception("feedback_id is required.")
    else:
        return await store.adelete_document(feedback.feedback_id)
