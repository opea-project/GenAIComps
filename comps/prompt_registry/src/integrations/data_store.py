# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import HTTPException

from comps.cores.storages.models import PromptCreate, PromptId
from comps.cores.storages.stores import get_store


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


def _preprocess(prompt: PromptCreate) -> dict:
    """Converts a PromptCreate object to a dictionary suitable for storage.

    Args:
        prompt (PromptCreate): The PromptCreate object to be converted.

    Returns:
        dict: A dictionary representation of the PromptCreate, ready for storage.
    """
    return {"prompt_text": prompt.prompt_text, "user": prompt.user}


async def save(prompt: PromptCreate):
    check_user_info(prompt)
    store = get_store(prompt.user)
    return await store.asave_document(_preprocess(prompt))


async def get(prompt: PromptId):
    check_user_info(prompt)
    store = get_store(prompt.user)
    if prompt.prompt_id is not None:
        return await store.aget_document_by_id(prompt.prompt_id)
    elif prompt.prompt_text:
        return await store.asearch(prompt.prompt_text)
    else:
        return await store.aget_documents_by_user(prompt.user)


async def delete(prompt: PromptId):
    check_user_info(prompt)
    store = get_store(prompt.user)
    if prompt.prompt_id is None:
        raise Exception("Prompt id is required.")
    else:
        return await store.adelete_document(prompt.prompt_id)
