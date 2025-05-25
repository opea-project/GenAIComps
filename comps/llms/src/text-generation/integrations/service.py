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
from comps.cores.proto.api_protocol import ChatCompletionRequest

from .template import ChatTemplate

logger = CustomLogger("opea_llm")

# Configure logger level based on LOGFLAG environment variable
if os.getenv("LOGFLAG", "False").lower() in ("true", "1", "yes"):
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
    """A specialized OPEA LLM component for interacting with TGI/vLLM or other OpenAI API-compatible services.

    - Handles formatting of input types (SearchedDoc, LLMParamsDoc, ChatCompletionRequest) for OpenAI-like API compatibility
    - Fields in input types are used to generate prompts and are therefore omitted in final openai api call. 
    - Allows provider-specific model inputs to pass through to the OpenAI API.

    Attributes:
        client: An instance of an OpenAI-compatible client (e.g., TGI/vLLM) for text generation
    """
    # Parameters to omit from openAI-like API calls 
    # The align_input method will format these fields into a prompt, and will omit these redundant fields from the final openai API call.
    OMIT_COMMON_PARAMS = {
        "chat_template",
        "documents",
    }

    OMIT_SEARCHDOC_PARAMS = OMIT_COMMON_PARAMS | {
        "initial_query",
        "retrieved_docs",
        "text",
    }

    OMIT_LLMPARAMS_PARAMS = OMIT_COMMON_PARAMS | {
        "query",
    }

    OMIT_CHATCOMPLETION_PARAMS = OMIT_COMMON_PARAMS  # No additional parameters to omit

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
        """Checks the health of the openAI compatible client.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """

        try:
            logger.debug(f"OpeaTextGenService: self.client.base_url: {self.client.base_url}")

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
        self, input: Union[LLMParamsDoc, ChatCompletionRequest, SearchedDoc], prompt_template, prompt_inputs
    ) -> dict:
        """Aligns different input types to a standardized format for API calls.
        
        The method will format inputs, and retain only those necessary for openai API call.

        Builds API parameters with the following considerations:
        1. Exclude None values since many OpenAI-compatible APIs don't accept null parameters
        2. Exclude parameters used internally for prompt generation (defined in OMIT_*_PARAMS)
        3. Allow provider-specific model parameters to pass through e.g. https://openrouter.ai/docs/use-cases/reasoning-tokens

        Args:
            input: The input request object (SearchedDoc, LLMParamsDoc, or ChatCompletionRequest)
            prompt_template: Optional template for formatting prompts
            prompt_inputs: Variables expected by the prompt template
            
        Returns:
            dict: The filtered parameters for the OpenAI API call
        """
        # These parameters are used to generate the prompts and are therefore omitted in final openai api call. 
        omit_params = (
            self.OMIT_SEARCHDOC_PARAMS if isinstance(input, SearchedDoc)
            else self.OMIT_LLMPARAMS_PARAMS if isinstance(input, LLMParamsDoc)
            else self.OMIT_CHATCOMPLETION_PARAMS
        )

        completion_params = {"model": MODEL_NAME}
        completion_params.update({k: v for k, v in vars(input).items() if v is not None and k not in omit_params})

        # Generate prompt based on input type and available context
        if isinstance(input, SearchedDoc):
            logger.debug("Processing SearchedDoc from retriever")
            prompt = input.initial_query
            if input.retrieved_docs:
                docs = [doc.text for doc in input.retrieved_docs]
                prompt = ChatTemplate.generate_rag_prompt(input.initial_query, docs, MODEL_NAME)
                logger.debug(f"[ SearchedDoc ] combined retrieved docs: {docs}")
        
        elif isinstance(input, LLMParamsDoc):
            logger.debug("[ LLMParamsDoc ] input from rerank microservice")
            prompt = input.query
            if prompt_template:
                if sorted(prompt_inputs) == ["context", "question"]:
                    prompt = prompt_template.format(question=input.query, context="\n".join(input.documents))
                elif prompt_inputs == ["question"]:
                    prompt = prompt_template.format(question=input.query)
                else:
                    logger.info(
                        f"[ LLMParamsDoc ] {prompt_template} not used, we only support 2 input variables ['question', 'context']"
                    )
            elif input.documents:
                prompt = ChatTemplate.generate_rag_prompt(input.query, input.documents, input.model)
        
        else:  # ChatCompletionRequest or regular request
            logger.debug("[ ChatCompletionRequest ] input in opea format")
            prompt = input.messages
            if prompt_template:
                if sorted(prompt_inputs) == ["context", "question"]:
                    prompt = prompt_template.format(question=input.messages, context="\n".join(input.documents))
                elif prompt_inputs == ["question"]:
                    prompt = prompt_template.format(question=input.messages)
                else:
                    logger.info(
                        f"[ ChatCompletionRequest ] {prompt_template} not used, we only support 2 input variables ['question', 'context']"
                    )
            elif input.documents:
                prompt = ChatTemplate.generate_rag_prompt(input.messages, input.documents, input.model)

            if isinstance(input, ChatCompletionRequest):
                completion_params["messages"] = prompt
            else:
                completion_params["prompt"] = prompt

        logger.debug(f"Filtered parameters: {completion_params}")

        return completion_params

    async def invoke(self, input: Union[LLMParamsDoc, ChatCompletionRequest, SearchedDoc]):
        """Invokes the TGI/vLLM LLM service to generate output for the provided input.

        Args:
            input (Union[LLMParamsDoc, ChatCompletionRequest, SearchedDoc]): The input text(s).
        """
        logger.debug(f"Input parameters:\n{pformat(vars(input), indent=2, width=120)}")
        logger.debug(f"Base URL: {self.client.base_url}")

        # Handle prompt template if present and valid
        try:
            prompt_template = PromptTemplate.from_template(input.chat_template) if not isinstance(input, SearchedDoc) and input.chat_template else None
            prompt_inputs = prompt_template.input_variables if prompt_template else None
        except (AttributeError, ValueError):
            prompt_template = None
            prompt_inputs = None
        
        completion_params = self.align_input(input, prompt_template, prompt_inputs)
        logger.debug(f"Formatted completion parameters:\n{pformat(completion_params, indent=2, width=120)}")

        if isinstance(input, ChatCompletionRequest) and not isinstance(input.messages, str):
            if input.messages[0]["role"] == "system":
                # Case 1: Message array already starts with a system message
                # If it contains a {context} placeholder, we fill it with available documents
                if "{context}" in input.messages[0]["content"]:
                    context = "" if input.documents is None or input.documents == [] else "\n".join(input.documents)
                    input.messages[0]["content"] = input.messages[0]["content"].format(context=context)
            elif prompt_template and prompt_inputs == ["context"] and input.documents:
                # Case 2: No system message yet, but we have a prompt template that expects context
                # We create a new system message with the template and add it at the start
                system_prompt = prompt_template.format(context="\n".join(input.documents))
                input.messages.insert(0, {"role": "system", "content": system_prompt})

            completion_params = self.align_input(input, prompt_template, prompt_inputs)
            chat_completion = await self.client.chat.completions.create(**completion_params)
            """TODO need validate following parameters for vllm
                logit_bias=input.logit_bias,
                logprobs=input.logprobs,
                top_logprobs=input.top_logprobs,
                service_tier=input.service_tier,
                tools=input.tools,
                tool_choice=input.tool_choice,
                parallel_tool_calls=input.parallel_tool_calls,"""
        else:
            # Handle regular completions
            _, completion_params = self.align_input(input, prompt_template, prompt_inputs)
            chat_completion = await self.client.completions.create(**completion_params)
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
