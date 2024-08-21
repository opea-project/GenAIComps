# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from enum import IntEnum
from typing import Any, Dict, List, Literal, Optional, Union

import shortuuid
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ServiceCard(BaseModel):
    object: str = "service"
    service_name: str
    description: str
    created: int = Field(default_factory=lambda: int(time.time()))
    owner: str = "opea"


class ServiceList(BaseModel):
    object: str = "list"
    data: List[ServiceCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    model: Optional[str] = "Intel/neural-chat-7b-v3-3"
    temperature: Optional[float] = 0.01
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 10
    n: Optional[int] = 1
    max_tokens: Optional[int] = 1024
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 1.03
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class AudioChatCompletionRequest(BaseModel):
    audio: str
    messages: Optional[
        Union[
            str,
            List[Dict[str, str]],
            List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
        ]
    ] = None
    model: Optional[str] = "Intel/neural-chat-7b-v3-3"
    temperature: Optional[float] = 0.01
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 10
    n: Optional[int] = 1
    max_tokens: Optional[int] = 1024
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 1.03
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


class TokenCheckRequestItem(BaseModel):
    model: str
    prompt: str
    max_tokens: int


class TokenCheckRequest(BaseModel):
    prompts: List[TokenCheckRequestItem]


class TokenCheckResponseItem(BaseModel):
    fits: bool
    tokenCount: int
    contextLength: int


class TokenCheckResponse(BaseModel):
    prompts: List[TokenCheckResponseItem]


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None
    encoding_format: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[Any]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    max_tokens: Optional[int] = 16
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    use_beam_search: Optional[bool] = False
    best_of: Optional[int] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]


class AudioQnaRequest(BaseModel):
    file: UploadFile = File(...)
    language: str = "auto"


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class ApiErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004


def create_error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(content=ErrorResponse(message=message, code=status_code), status_code=status_code.value)


def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )

    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.top_k is not None and (request.top_k > -1 and request.top_k < 1):
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_k} is out of Range. Either set top_k to -1 or >=1.",
        )
    if request.stop is not None and (not isinstance(request.stop, str) and not isinstance(request.stop, list)):
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )

    return None


class Hyperparameters(BaseModel):
    batch_size: Optional[Union[Literal["auto"], int]] = "auto"
    """Number of examples in each batch.
    A larger batch size means that model parameters are updated less frequently, but with lower variance.
    """

    learning_rate_multiplier: Optional[Union[Literal["auto"], float]] = "auto"
    """Scaling factor for the learning rate.
    A smaller learning rate may be useful to avoid overfitting.
    """

    n_epochs: Optional[Union[Literal["auto"], int]] = "auto"
    """The number of epochs to train the model for.
    An epoch refers to one full cycle through the training dataset. "auto" decides
    the optimal number of epochs based on the size of the dataset. If setting the
    number manually, we support any number between 1 and 50 epochs.
    """


class FineTuningJobWandbIntegration(BaseModel):
    project: str
    """The name of the project that the new run will be created under."""

    entity: Optional[str] = None
    """The entity to use for the run.
    This allows you to set the team or username of the WandB user that you would
    like associated with the run. If not set, the default entity for the registered
    WandB API key is used.
    """

    name: Optional[str] = None
    """A display name to set for the run.
    If not set, we will use the Job ID as the name.
    """

    tags: Optional[List[str]] = None
    """A list of tags to be attached to the newly created run.
    These tags are passed through directly to WandB. Some default tags are generated
    by OpenAI: "openai/finetune", "openai/{base-model}", "openai/{ftjob-abcdef}".
    """


class FineTuningJobWandbIntegrationObject(BaseModel):
    type: Literal["wandb"]
    """The type of the integration being enabled for the fine-tuning job."""

    wandb: FineTuningJobWandbIntegration
    """The settings for your integration with Weights and Biases.
    This payload specifies the project that metrics will be sent to. Optionally, you
    can set an explicit display name for your run, add tags to your run, and set a
    default entity (team, username, etc) to be associated with your run.
    """


class FineTuningJobsRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/fine-tuning/create
    model: str
    """The name of the model to fine-tune."""

    training_file: str
    """The ID of an uploaded file that contains training data."""

    hyperparameters: Optional[Hyperparameters] = None
    """The hyperparameters used for the fine-tuning job."""

    suffix: Optional[str] = None
    """A string of up to 64 characters that will be added to your fine-tuned model name."""

    validation_file: Optional[str] = None
    """The ID of an uploaded file that contains validation data."""

    integrations: Optional[List[FineTuningJobWandbIntegrationObject]] = None
    """A list of integrations to enable for your fine-tuning job."""

    seed: Optional[str] = None


class Error(BaseModel):
    code: str
    """A machine-readable error code."""

    message: str
    """A human-readable error message."""

    param: Optional[str] = None
    """The parameter that was invalid, usually `training_file` or `validation_file`.
    This field will be null if the failure was not parameter-specific.
    """


class FineTuningJob(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/fine-tuning/object
    id: str
    """The object identifier, which can be referenced in the API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the fine-tuning job was created."""

    error: Optional[Error] = None
    """For fine-tuning jobs that have `failed`, this will contain more information on
    the cause of the failure."""

    fine_tuned_model: Optional[str] = None
    """The name of the fine-tuned model that is being created.
    The value will be null if the fine-tuning job is still running.
    """

    finished_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the fine-tuning job was finished.
    The value will be null if the fine-tuning job is still running.
    """

    hyperparameters: Hyperparameters
    """The hyperparameters used for the fine-tuning job.
    See the [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)
    for more details.
    """

    model: str
    """The base model that is being fine-tuned."""

    object: Literal["fine_tuning.job"] = "fine_tuning.job"
    """The object type, which is always "fine_tuning.job"."""

    organization_id: Optional[str] = None
    """The organization that owns the fine-tuning job."""

    result_files: List[str] = None
    """The compiled results file ID(s) for the fine-tuning job.
    You can retrieve the results with the
    [Files API](https://platform.openai.com/docs/api-reference/files/retrieve-contents).
    """

    status: Literal["validating_files", "queued", "running", "succeeded", "failed", "cancelled"]
    """The current status of the fine-tuning job, which can be either
    `validating_files`, `queued`, `running`, `succeeded`, `failed`, or `cancelled`."""

    trained_tokens: Optional[int] = None
    """The total number of billable tokens processed by this fine-tuning job.
    The value will be null if the fine-tuning job is still running.
    """

    training_file: str
    """The file ID used for training.
    You can retrieve the training data with the
    [Files API](https://platform.openai.com/docs/api-reference/files/retrieve-contents).
    """

    validation_file: Optional[str] = None
    """The file ID used for validation.
    You can retrieve the validation results with the
    [Files API](https://platform.openai.com/docs/api-reference/files/retrieve-contents).
    """

    integrations: Optional[List[FineTuningJobWandbIntegrationObject]] = None
    """A list of integrations to enable for this fine-tuning job."""

    seed: Optional[int] = None
    """The seed used for the fine-tuning job."""

    estimated_finish: Optional[int] = None
    """The Unix timestamp (in seconds) for when the fine-tuning job is estimated to
    finish.
    The value will be null if the fine-tuning job is not running.
    """


class FineTuningJobIDRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/fine-tuning/retrieve
    # https://platform.openai.com/docs/api-reference/fine-tuning/cancel
    fine_tuning_job_id: str
    """The ID of the fine-tuning job."""


class FineTuningJobListRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/fine-tuning/list
    after: Optional[str] = None
    """Identifier for the last job from the previous pagination request."""

    limit: Optional[int] = 20
    """Number of fine-tuning jobs to retrieve."""


class FineTuningJobList(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/fine-tuning/list
    object: str = "list"
    """The object type, which is always "list".
    This indicates that the returned data is a list of fine-tuning jobs.
    """

    data: List[FineTuningJob]
    """A list containing FineTuningJob objects."""

    has_more: bool
    """Indicates whether there are more fine-tuning jobs beyond the current list.
    If true, additional requests can be made to retrieve more jobs.
    """
