# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0

import asyncio
import logging
import os
from pprint import pformat
from typing import Union

from fastapi.responses import StreamingResponse
from langchain_core.prompts import PromptTemplate
from openai import AsyncOpenAI

from comps import CustomLogger, LLMParamsDoc, OpeaComponent, OpeaComponentRegistry, SearchedDoc, ServiceType
from comps.cores.mega.utils import ConfigError, get_access_token, load_model_configs
from comps.cores.proto.api_protocol import ALLOWED_CHATCOMPLETION_ARGS, ALLOWED_COMPLETION_ARGS, ChatCompletionRequest

from .template import ChatTemplate

logger = CustomLogger("opea_llm")

# Configure advanced logging based on LOGFLAG environment variable
logflag = os.getenv("LOGFLAG", "False").lower() in ("true", "1", "yes")
if logflag:
    logger.logger.setLevel(logging.DEBUG)
else:
    logger.logger.setLevel(logging.INFO)

# Environment variables
MODEL_NAME = os.getenv("LLM_MODEL_ID")
MODEL_CONFIGS = os.getenv("MODEL_CONFIGS")
DEFAULT_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8080")
TOKEN_URL = os.getenv("TOKEN_URL")
CLIENTID = os.getenv("CLIENTID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

# Validate and Load the models config if MODEL_CONFIGS is not null
configs_map = {}
if MODEL_CONFIGS:
    try:
        configs_map = load_model_configs(MODEL_CONFIGS)
    except ConfigError as e:
        logger.error(f"Failed to load model configurations: {e}")
        raise ConfigError(f"Failed to load model configurations: {e}")


def get_llm_endpoint():
    if not MODEL_CONFIGS:
        return DEFAULT_ENDPOINT
    try:
        return configs_map.get(MODEL_NAME).get("endpoint")
    except ConfigError as e:
        logger.error(f"Input model {MODEL_NAME} not present in model_configs. Error {e}")
        raise ConfigError(f"Input model {MODEL_NAME} not present in model_configs")


@OpeaComponentRegistry.register("OpeaTextGenService")
class OpeaTextGenService(OpeaComponent):
    """A specialized OPEA LLM component derived from OpeaComponent for interacting with TGI/vLLM services based on OpenAI API.

    Attributes:
        client (TGI/vLLM): An instance of the TGI/vLLM client for text generation.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.LLM.name.lower(), description, config)
        self.client = self._initialize_client()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaTextGenService health check failed.")

    def _initialize_client(self) -> AsyncOpenAI:
        """Initializes the AsyncOpenAI."""
        access_token = (
            get_access_token(TOKEN_URL, CLIENTID, CLIENT_SECRET) if TOKEN_URL and CLIENTID and CLIENT_SECRET else None
        )
        headers = {}
        if access_token:
            headers = {"Authorization": f"Bearer {access_token}"}
        llm_endpoint = get_llm_endpoint()
        return AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=llm_endpoint + "/v1", timeout=600, default_headers=headers)

    def check_health(self) -> bool:
        """Checks the health of the TGI/vLLM LLM service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """

        try:

            async def send_simple_request():
                response = await self.client.completions.create(model=MODEL_NAME, prompt="How are you?", max_tokens=4)
                return response

            response = asyncio.run(send_simple_request())
            return response is not None
        except Exception as e:
            logger.error(e)
            logger.error("Health check failed")
            return False

    def align_input(
        self, input: Union[LLMParamsDoc, ChatCompletionRequest, SearchedDoc], prompt_template, input_variables
    ):
        """Aligns different input types to a standardized chat completion format.

        Args:
            input: SearchedDoc, LLMParamsDoc, or ChatCompletionRequest
            prompt_template: Optional template for formatting prompts
            input_variables: Variables expected by the prompt template
        """
        if isinstance(input, SearchedDoc):
            logger.debug(f"Processing SearchedDoc input from retriever microservice:\n{pformat(vars(input), indent=2)}")
            prompt = input.initial_query
            if input.retrieved_docs:
                docs = [doc.text for doc in input.retrieved_docs]
                logger.debug(f"Retrieved documents:\n{pformat(docs, indent=2)}")
                prompt = ChatTemplate.generate_rag_prompt(input.initial_query, docs, MODEL_NAME)
                logger.debug(f"Generated RAG prompt:\n{prompt}")

                # Convert to ChatCompletionRequest with default parameters
                new_input = ChatCompletionRequest(messages=prompt)
                logger.debug(f"Final converted input:\n{pformat(vars(new_input), indent=2)}")

                return prompt, new_input

        elif isinstance(input, LLMParamsDoc):
            logger.debug(f"Processing LLMParamsDoc input from rerank microservice:\n{pformat(vars(input), indent=2)}")
            prompt = input.query
            if prompt_template:
                if sorted(input_variables) == ["context", "question"]:
                    prompt = prompt_template.format(question=input.query, context="\n".join(input.documents))
                elif input_variables == ["question"]:
                    prompt = prompt_template.format(question=input.query)
                else:
                    logger.warning(
                        f"Prompt template not used - unsupported variables. Template: {prompt_template}\nOnly ['question', 'context'] or ['question'] are supported"
                    )
            else:
                if input.documents:
                    # use rag default template
                    prompt = ChatTemplate.generate_rag_prompt(input.query, input.documents, input.model)

            # convert to unified OpenAI /v1/chat/completions format
            new_input = ChatCompletionRequest(
                messages=prompt,
                max_tokens=input.max_tokens,
                top_p=input.top_p,
                stream=input.stream,
                frequency_penalty=input.frequency_penalty,
                temperature=input.temperature,
            )

            return prompt, new_input

        else:
            logger.debug(f"Processing ChatCompletionRequest input:\n{pformat(vars(input), indent=2)}")

            prompt = input.messages
            if prompt_template:
                if sorted(input_variables) == ["context", "question"]:
                    prompt = prompt_template.format(question=input.messages, context="\n".join(input.documents))
                elif input_variables == ["question"]:
                    prompt = prompt_template.format(question=input.messages)
                else:
                    logger.info(
                        f"[ ChatCompletionRequest ] {prompt_template} not used, we only support 2 input variables ['question', 'context']"
                    )
            else:
                if input.documents:
                    # use rag default template
                    prompt = ChatTemplate.generate_rag_prompt(input.messages, input.documents, input.model)

            return prompt, input

    async def invoke(self, input: Union[LLMParamsDoc, ChatCompletionRequest, SearchedDoc]):
        """Invokes the TGI/vLLM LLM service to generate output for the provided input.

        Args:
            input (Union[LLMParamsDoc, ChatCompletionRequest, SearchedDoc]): The input text(s).
        """

        prompt_template = None
        input_variables = None
        if not isinstance(input, SearchedDoc) and input.chat_template:
            prompt_template = PromptTemplate.from_template(input.chat_template)
            input_variables = prompt_template.input_variables

        if isinstance(input, ChatCompletionRequest) and not isinstance(input.messages, str):
            logger.debug("[ ChatCompletionRequest ] input in opea format")

            if input.messages[0]["role"] == "system":
                if "{context}" in input.messages[0]["content"]:
                    if input.documents is None or input.documents == []:
                        input.messages[0]["content"].format(context="")
                    else:
                        input.messages[0]["content"].format(context="\n".join(input.documents))
            else:
                if prompt_template:
                    system_prompt = prompt_template
                    if input_variables == ["context"]:
                        system_prompt = prompt_template.format(context="\n".join(input.documents))
                    else:
                        logger.info(
                            f"[ ChatCompletionRequest ] {prompt_template} not used, only support 1 input variables ['context']"
                        )

                    input.messages.insert(0, {"role": "system", "content": system_prompt})

            # Create input params directly from input object attributes
            input_params = {**vars(input), "model": MODEL_NAME}
            filtered_params = self._filter_api_params(input_params, ALLOWED_CHATCOMPLETION_ARGS)
            logger.debug(f"Filtered chat completion parameters:\n{pformat(filtered_params, indent=2)}")
            chat_completion = await self.client.chat.completions.create(**filtered_params)
            """TODO need validate following parameters for vllm
                logit_bias=input.logit_bias,
                logprobs=input.logprobs,
                top_logprobs=input.top_logprobs,
                service_tier=input.service_tier,
                tools=input.tools,
                tool_choice=input.tool_choice,
                parallel_tool_calls=input.parallel_tool_calls,"""
        else:
            prompt, input = self.align_input(input, prompt_template, input_variables)
            input_params = {**vars(input), "model": MODEL_NAME, "prompt": prompt}
            filtered_params = self._filter_api_params(input_params, ALLOWED_COMPLETION_ARGS)
            logger.debug(f"Filtered completion parameters:\n{pformat(filtered_params, indent=2)}")
            chat_completion = await self.client.completions.create(**filtered_params)
            """TODO need validate following parameters for vllm
                best_of=input.best_of,
                logit_bias=input.logit_bias,
                logprobs=input.logprobs,"""

        if input.stream:

            async def stream_generator():
                async for c in chat_completion:
                    logger.debug(c)
                    chunk = c.model_dump_json()
                    if chunk not in ["<|im_end|>", "<|endoftext|>"]:
                        yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            logger.debug(chat_completion)
            return chat_completion

    def _filter_api_params(self, input_params: dict, allowed_args: tuple) -> dict:
        """Filters input parameters to only include allowed non-None arguments.

        Only allow allowed args, and and filter non-None default arguments because
        some open AI-like APIs e.g. OpenRouter.ai will disallow None parameters.
        Works for both chat completion and regular completion API calls.

        Args:
            input_params: Dictionary of input parameters
            allowed_args: Tuple of allowed argument names

        Returns:
            Filtered dictionary containing only allowed non-None arguments
        """
        return {arg: input_params[arg] for arg in allowed_args if arg in input_params and input_params[arg] is not None}
