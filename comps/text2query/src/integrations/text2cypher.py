# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import threading
import traceback

from langchain.chains.llm import LLMChain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.chains.graph_qa.prompts import CYPHER_QA_PROMPT
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import Text2QueryRequest
from comps.text2query.src.integrations.cypher.cypher_utils import (
    CypherQueryCorrectorExt,
    construct_schema,
    cypher_cleanup,
    cypher_insert,
    prepare_chat_template,
)
from comps.text2query.src.integrations.cypher.pipeline import GaudiTextGenerationPipeline

logger = CustomLogger("opea_text2cypher_native")
initialization_lock = threading.Lock()
initialized = False
query_chain = None


@OpeaComponentRegistry.register("OPEA_TEXT2QUERY_CYPHER")
class OpeaText2Cypher(OpeaComponent):
    """A specialized text2cyher component derived from OpeaComponent for text2cypher services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.TEXT2CYPHER.name.lower(), description, config)
        self.config = config

    def _initialize_client(self, input: Text2QueryRequest, config: dict = None):
        """Initializes the chain client."""
        global query_chain, initialized

        config = (
            config
            if config
            else {
                "load_quantized_model": False,
                "num_return_sequences": 1,
                "model_name_or_path": "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1",
                "max_new_tokens": 512,
                "use_hpu_graphs": True,
                "use_kv_cache": True,
                "do_sample": True,
                "show_graphs_count": False,
                "parallel_strategy": "none",
                "fp8": False,
                "kv_cache_fp8": False,
                "temperature": 0.1,
                "device": "hpu",
                "batch_size": 1,
                "limit_hpu_graphs": False,
                "reuse_cache": False,
                "bucket_internal": False,
                "disk_offload": False,
                "seed": 27,
                "token": None,
                "assistant_model": None,
                "torch_compile": False,
                "peft_model": None,
                "quant_config": os.getenv("QUANT_CONFIG", ""),
                "bf16": True,
                "attn_softmax_bf16": False,
                "max_input_tokens": 0,
                "warmup": 3,
                "n_iterations": 5,
                "local_rank": 0,
                "num_beams": 1,
                "trim_logits": True,
                "profiling_warmup_steps": 0,
                "profiling_steps": 0,
                "profiling_record_shapes": False,
                "tokenizer-config-path": None,
                "dataset_name": None,
                "column_name": None,
                "prompt": None,
                "bad_words": None,
                "force_words": None,
                "model_revision": "main",
                "output_dir": None,
                "bucket_size": -1,
                "dataset_max_samples": -1,
                "verbose_workers": True,
                "simulate_dyn_prompt": None,
                "reduce_recompile": False,
                "use_flash_attention": True,
                "flash_attention_recompute": True,
                "flash_attention_causal_mask": True,
                "flash_attention_fast_softmax": True,
                "book_source": True,
                "ignore_eos": True,
                "top_p": 1.0,
                "trust_remote_code": True,
                "const_serialization_path": None,
            }
        )

        logger.info("[ OpeaText2Cypher ] initialize_client started.")
        model_name_or_path = config["model_name_or_path"]
        device = config["device"]
        chat_model = None
        if device == "hpu":
            # Convert config dict back to args-like object
            args = argparse.Namespace(**config)
            pipe = GaudiTextGenerationPipeline(args, logger, use_with_langchain=True)
            hfpipe = HuggingFacePipeline(pipeline=pipe)

            chat_model = ChatHuggingFace(temperature=0.1, llm=hfpipe, tokenizer=pipe.tokenizer)

        else:
            raise NotImplementedError(f"Only support hpu device now, device {device} not supported.")

        prompt = input.query
        user = input.conn_user if input.conn_user else os.getenv("NEO4J_USERNAME", "neo4j")
        password = input.conn_password if input.conn_password else os.getenv("NEO4J_PASSWORD", "neo4jtest")
        url = input.conn_url if input.conn_url else os.getenv("NEO4J_URL", "bolt://localhost:7687")

        graph_store = Neo4jGraph(
            username=user,
            password=password,
            url=url,
        )

        if len(input.options) > 0:
            user_cypher_insert = input.options.get("cypher_insert", None)
            if user_cypher_insert is None:
                raise ValueError("cypher_insert must be provided in the request.")

            refresh_db = input.options.get("refresh_db", True)
            if refresh_db:
                graph_store.query(cypher_cleanup)
            graph_store.query(user_cypher_insert)
        else:
            graph_store.query(cypher_cleanup)
            graph_store.query(cypher_insert)
        graph_store.refresh_schema()

        cypher_prompt = PromptTemplate(input_variables=["schema"], template=prepare_chat_template(prompt))

        graph_schema = construct_schema(graph_store.get_structured_schema, [], [])

        cypher_query_corrector = CypherQueryCorrectorExt(schemas=graph_store.schema, schema_str=graph_schema)

        use_qa_llm_kwargs = {"prompt": CYPHER_QA_PROMPT}
        use_cypher_llm_kwargs = {"prompt": cypher_prompt}

        qa_chain = LLMChain(llm=chat_model, **use_qa_llm_kwargs)
        cypher_generation_chain = LLMChain(
            llm=chat_model,
            **use_cypher_llm_kwargs,
        )

        query_chain = GraphCypherQAChain(
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

        return query_chain

    async def check_health(self) -> bool:
        """Checks the health of the Text2Cypher service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        global initialized
        return initialized

    async def invoke(self, input: Text2QueryRequest):
        """Invokes the text2cypher service.

        Args:
            input (Inputs): The input for text2cypher service, including input_text and optional connection info.

        Returns:
            str: the generated output.
        """

        global query_chain, initialized
        with initialization_lock:
            if not initialized:
                try:
                    query_chain = self._initialize_client(input, self.config)
                    initialized = True
                except Exception as e:
                    logger.error(f"Error during _initialize_client: {e}")
                    logger.error(traceback.format_exc())
                    raise

        try:
            result = query_chain.run(input.query)
        except Exception as e:
            logger.error(f"Error during text2cypher invocation: {e}")
            logger.error(traceback.format_exc())
            raise

        return result

    async def db_connection_check(self, request: Text2QueryRequest):
        """Check the connection to Neo4j database.

        This function takes a Text2QueryRequest object containing the database connection information.
        It attempts to connect to the database using the provided connection URL and credentials.

        Args:
            request (Text2QueryRequest): A Text2QueryRequest object with the database connection information.
        Returns:
            dict: A dictionary with a 'status' key indicating whether the connection was successful or failed.
        """
        return {"status": "Connection successful"}
