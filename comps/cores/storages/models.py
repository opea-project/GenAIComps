# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Optional

from pydantic import BaseModel, Field

from comps.cores.proto.api_protocol import ChatCompletionRequest

#################################################################
# NOTE: Below are the data models for the different storage     #
#       components which were previously in the respective      #
#       microservices: ChatHistory, FeedbackManagement, and     #
#       PromptRegistry. These models are now consolidated here  #
#       for reusability and maintainability.                    #
#################################################################

################
# Chat History #
################


class ChatMessage(BaseModel):
    data: ChatCompletionRequest
    first_query: Optional[str] = None
    id: Optional[str] = None


class ChatId(BaseModel):
    user: str
    id: Optional[str] = None


#######################
# Feedback Management #
#######################


class FeedbackData(BaseModel):
    """This class represents the data model of FeedbackData collected to store in database.".

    Attributes:
        is_thumbs_up (bool): True if the response is satisfy, False otherwise.
        rating: (int)[Optional]: Score rating. Range from 0 (bad rating) to 5(good rating).
        comment (str)[Optional]: Comment given for response.
    """

    is_thumbs_up: bool
    rating: Annotated[Optional[int], Field(ge=0, le=5)] = None
    comment: Optional[str] = None


class ChatFeedback(BaseModel):
    """This class represents the model for chat to collect FeedbackData together with ChatCompletionRequest data to store in database.

    Attributes:
        chat_data (ChatCompletionRequest): ChatCompletionRequest object containing chat data to be stored.
        feedback_data (FeedbackData): FeedbackData object containing feedback data for chat to be stored.
        chat_id (str)[Optional]: The chat_id associated to the chat to be store together with feedback data.
        feedback_id (str)[Optional]: The feedback_id of feedback data to be retrieved from database.
    """

    chat_data: ChatCompletionRequest
    feedback_data: FeedbackData
    chat_id: Optional[str] = None
    feedback_id: Optional[str] = None


class FeedbackId(BaseModel):
    """This class represent the data model for retrieve feedback data stored in database.

    Attributes:
        user (str): The user of the requested feedback data.
        feedback_id (str): The feedback_id of feedback data to be retrieved from database.
    """

    user: str
    feedback_id: Optional[str] = None


###################
# Prompt Registry #
###################


class PromptCreate(BaseModel):
    """This class represents the data model for creating and storing a new prompt in the database.

    Attributes:
        prompt_text (str): The text content of the prompt.
        user (str): The user or creator of the prompt.
    """

    prompt_text: str
    user: str


class PromptId(BaseModel):
    """This class represent the data model for retrieve prompt stored in database.

    Attributes:
        user (str): The user of the requested prompt.
        prompt_id (str): The prompt_id of prompt to be retrieved from database.
    """

    user: str
    prompt_id: Optional[str] = None
    prompt_text: Optional[str] = None
