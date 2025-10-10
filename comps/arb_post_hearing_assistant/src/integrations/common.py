# Copyright (C) 2025 Zensar Technologies Private Ltd.
# SPDX-License-Identifier: Apache-2.0

import os

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate

from comps import CustomLogger, GeneratedDoc, OpeaComponent, ServiceType
from comps.cores.mega.utils import ConfigError, load_model_configs
from comps.cores.proto.api_protocol import ArbPostHearingAssistantChatCompletionRequest

from .template import arbitratory_template

logger = CustomLogger("arb_post_hearing_assistant_tgi_microservice")
logflag = os.getenv("LOGFLAG", False)

# Environment variables
MODEL_NAME = os.getenv("LLM_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
MODEL_CONFIGS = os.getenv("MODEL_CONFIGS")
MAX_INPUT_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", "4096"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "2048"))

if os.getenv("LLM_ENDPOINT") is not None:
    DEFAULT_ENDPOINT = os.getenv("LLM_ENDPOINT")
elif os.getenv("TGI_LLM_ENDPOINT") is not None:
    DEFAULT_ENDPOINT = os.getenv("TGI_LLM_ENDPOINT")
elif os.getenv("vLLM_ENDPOINT") is not None:
    DEFAULT_ENDPOINT = os.getenv("vLLM_ENDPOINT")
else:
    DEFAULT_ENDPOINT = "http://localhost:8080"


def get_llm_endpoint():
    if not MODEL_CONFIGS:
        return DEFAULT_ENDPOINT
    else:
        # Validate and Load the models config if MODEL_CONFIGS is not null
        configs_map = {}
        try:
            configs_map = load_model_configs(MODEL_CONFIGS)
        except ConfigError as e:
            logger.error(f"Failed to load model configurations: {e}")
            raise ConfigError(f"Failed to load model configurations: {e}")
        try:
            return configs_map.get(MODEL_NAME).get("endpoint")
        except ConfigError as e:
            logger.error(f"Input model {MODEL_NAME} not present in model_configs. Error {e}")
            raise ConfigError(f"Input model {MODEL_NAME} not present in model_configs")


class OpeaArbPostHearingAssistant(OpeaComponent):
    """A specialized OPEA ArbPostHearingAssistant component derived from OpeaComponent.

    Attributes:
        client (TGI/vLLM): An instance of the TGI/vLLM client for text generation.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.ARB_POST_HEARING_ASSISTANT.name.lower(), description, config)
        self.llm_endpoint = get_llm_endpoint()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaArbPostHearingAssistant health check failed.")

    async def generate(self, input: ArbPostHearingAssistantChatCompletionRequest, client):
        """Invokes the TGI/vLLM LLM service to generate summarization for the provided input.

        Args:
            input (ArbPostHearingAssistantChatCompletionRequest): The input text(s).
            client: TGI/vLLM based client
        """
        ### get input text
        message = None
        if isinstance(input.messages, str):
            message = input.messages
        if message is None:
            logger.error("Don't receive any input text, exit!")
            return GeneratedDoc(text=None, prompt=None)

        ## Prompt
        PROMPT = PromptTemplate.from_template(arbitratory_template)

        docs = [Document(page_content=message)]

        llm_chain = load_summarize_chain(llm=client, prompt=PROMPT)
        response = await llm_chain.ainvoke(docs)
        output_text = response["output_text"]

        if logflag:
            logger.info("\n\noutput_text:")
            logger.info(output_text)

        return GeneratedDoc(text=output_text, prompt=message)
