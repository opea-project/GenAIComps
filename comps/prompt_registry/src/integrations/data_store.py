# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import HTTPException

from comps.cores.storages.models import PromptCreate, PromptId
from comps.cores.storages.stores import get_store, postget, remove_db_private_cols


def check_user_info(prompt: PromptCreate | PromptId):
    """Checks if the user information is provided in the document.

    Args:
        document (ChatFeedback|FeedbackId): The document to be checked.

    Raises:
        HTTPException: If the user information is missing.
    """
    user = prompt.user
    if user is None or (isinstance(user, str) and user.strip() == ""):
        raise HTTPException(status_code=400, detail="User information is required but not provided")


def _prepersist(prompt: PromptCreate) -> dict:
    """Converts a PromptCreate object to a dictionary suitable for storage.

    Args:
        prompt (PromptCreate): The PromptCreate object to be converted.

    Returns:
        dict: A dictionary representation of the PromptCreate, ready for storage.
    """
    return {"prompt_text": prompt.prompt_text, "user": prompt.user}


def _post_getby_id(rs: dict) -> str:
    """Post-processes a single document retrieved by ID.

    Args:
        rs (dict): The document dictionary retrieved from storage.

    Returns:
        str: prompt_text.
    """
    return rs.get("prompt_text", None)


def _postget(rss: list) -> list:
    """Post-processes a list of documents by removing the ID column from each document.

    Args:
        rss (list): List of document dictionaries retrieved from storage.

    Returns:
        list: List of document dictionaries with ID columns removed.
    """
    for rs in rss:
        rs = remove_db_private_cols(rs)
    return rss


def _postsearch(rss: list) -> list:
    return [postget("id", doc) for doc in rss]


async def save(prompt: PromptCreate):
    """Saves a prompt to the data store after validating user information.

    Args:
        prompt (PromptCreate): The prompt object to be saved.

    Returns:
        The result of the save operation from the underlying storage.

    Raises:
        HTTPException: If user information validation fails.
    """
    check_user_info(prompt)
    store = get_store(prompt.user)
    return await store.asave_document(_prepersist(prompt))


async def get(prompt: PromptId):
    """Retrieves prompt(s) from the data store based on the provided criteria.

    Args:
        prompt (PromptId): The prompt identifier object containing search criteria.

    Returns:
        dict or list: A single prompt dictionary if searching by ID,
                     or a list of prompt dictionaries if searching by text or user.

    Raises:
        HTTPException: If user information validation fails.
    """
    check_user_info(prompt)
    store = get_store(prompt.user)
    if prompt.prompt_id is not None:
        rs = await store.aget_document_by_id(prompt.prompt_id)
        return _post_getby_id(rs)
    elif prompt.prompt_text:
        rss = await store.asearch_by_keyword(keyword=prompt.prompt_text, max_results=5, fields=["prompt_text", "user"])
        return _postsearch(rss)
    else:
        rss = await store.asearch(key="user", value=prompt.user)
        return _postget(rss)


async def delete(prompt: PromptId):
    """Deletes a prompt from the data store by its ID.

    Args:
        prompt (PromptId): The prompt identifier object containing the prompt ID to delete.

    Returns:
        The result of the delete operation from the underlying storage.

    Raises:
        HTTPException: If user information validation fails.
        Exception: If prompt_id is not provided.
    """
    check_user_info(prompt)
    store = get_store(prompt.user)
    if prompt.prompt_id is None:
        raise Exception("Prompt id is required.")
    else:
        return await store.adelete_document(prompt.prompt_id)
