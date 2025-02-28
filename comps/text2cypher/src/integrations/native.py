# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import sys
import time
from string import Template
from typing import Annotated, Any, Dict, List, Optional, Union

from langchain.chains.llm import LLMChain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.chains.graph_qa.prompts import CYPHER_QA_PROMPT
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# from langchain_community.chat_models import ChatGaudi
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline
from pydantic import BaseModel, Field
from pyprojroot import here
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.text2cypher.src.integrations.cypher_utils import (
    CypherQueryCorrector2,
    construct_schema,
    cypher_cleanup,
    cypher_insert,
    prepare_chat_template,
)

# from llama_index.core.indices.property_graph import LLMSynonymRetriever, VectorContextRetriever
# from load_llm import load_llm
from comps.text2cypher.src.integrations.pipeline import GaudiTextGenerationPipeline

# from comps.text2cypher.src.integrations.gaudiutils import initialize_model, setup_parser


logger = CustomLogger("native")

class Neo4jConnection(BaseModel):
    user: Annotated[str, Field(min_length=1)]
    password: Annotated[str, Field(min_length=1)]
    url: Annotated[str, Field(min_length=1)]


class Input(BaseModel):
    input_text: str
    conn_str: Optional[Neo4jConnection] = None


@OpeaComponentRegistry.register("OPEA_TEXT2CYPHER")
class OpeaText2Cypher(OpeaComponent):
    """A specialized text2cyher component derived from OpeaComponent for text2cypher services.

    Attributes:
        client (AsyncInferenceClient): An instance of the async client for cypher generation.
        model_name (str): The name of the embedding model used.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.TEXT2CYPHER.name.lower(), description, config)

        # initialize model and tokenizer
        model_name_or_path = config["model_name_or_path"]
        device = config["device"]
        self.chat_model = None
        if device == "hpu":
            # Convert config dict back to args-like object
            args = argparse.Namespace(**config)
            pipe = GaudiTextGenerationPipeline(args, logger, use_with_langchain=True)
            hfpipe = HuggingFacePipeline(pipeline=pipe)

            self.chat_model = ChatHuggingFace(temperature=0.1, llm=hfpipe, tokenizer=pipe.tokenizer)

        # elif device == "cpu":
        else:
            raise NotImplementedError(f"Only support hpu device now, device {device} not supported.")
        logger.info("model initialized.")

    async def invoke(self, input: Input):
        """Invokes the text2cypher service.

        Args:
            input (Inputs): The input for text2cypher service, including input_text and optional connection info.

        Returns:
            str: the generated output.
        """
        prompt = input.input_text
        user = input.conn_str.user
        password = input.conn_str.password
        url = input.conn_str.url

        graph_store = Neo4jGraph(
            username=user,
            password=password,
            url=url,
        )

        graph_store.query(cypher_cleanup)
        graph_store.query(cypher_insert)
        logger.info(f"Graph has been built with the following graph schema: {graph_store.schema}")

        cypher_prompt = PromptTemplate(input_variables=["schema"], template=prepare_chat_template(prompt))

        graph_schema = construct_schema(graph_store.get_structured_schema, [], [])

        cypher_query_corrector = CypherQueryCorrector2(schemas=graph_store.schema, schema_str=graph_schema)

        use_qa_llm_kwargs = {"prompt": CYPHER_QA_PROMPT}
        use_cypher_llm_kwargs = {"prompt": cypher_prompt}

        qa_chain = LLMChain(llm=self.chat_model, **use_qa_llm_kwargs)
        cypher_generation_chain = LLMChain(
            llm=self.chat_model,
            **use_cypher_llm_kwargs,
        )

        chain = GraphCypherQAChain(
            graph=graph_store,
            graph_schema=graph_schema,
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
            cypher_query_corrector=cypher_query_corrector,
            cypher_llm_kwargs={"prompt": cypher_prompt},
            verbose=True,
            return_intermediate_steps=True,
            return_direct=True,
            allow_dangerous_requests=True,
        )

        start_time = time.time()
        result = chain.run(prompt)
        end_time = time.time()
        latency = end_time - start_time

        logger.info(f"Latency: {latency:.2f} seconds.")
        logger.info(f"result: {result}")
        return result

    def check_health(self) -> bool:
        return True
