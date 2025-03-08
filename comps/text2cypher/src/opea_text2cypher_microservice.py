# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.text2cypher.src.integrations.gaudiutils import setup_parser
from comps.text2cypher.src.integrations.native import Input, Neo4jConnection, OpeaText2Cypher

logger = CustomLogger("opea_text2cypher_microservice")


@register_microservice(
    name="opea_service@text2cypher",
    service_type=ServiceType.TEXT2CYPHER,
    endpoint="/v1/text2cypher",
    host="0.0.0.0",
    port=9097,
)
@register_statistics(names=["opea_service@text2cypher"])
async def text2cypher(input: Input):
    start = time.time()
    try:
        # Use the loader to invoke the active component
        results = await loader.invoke(input)
        statistics_dict["opea_service@text2cypher"].append_latency(time.time() - start, None)
        return results
    except Exception as e:
        logger.error(f"Error during text2cypher invocation: {e}")
        logger.error("An error occurred:", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)

    args.load_quantized_model = False
    args.num_return_sequences = 1
    args.model_name_or_path = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
    args.max_new_tokens = 512
    args.use_hpu_graphs = True
    args.use_kv_cache = True
    args.do_sample = True
    args.show_graphs_count = False
    args.parallel_strategy = "none"
    args.fp8 = False
    args.kv_cache_fp8 = False
    args.temperature = 0.1
    args.device = "hpu"
    args.batch_size = 1
    args.limit_hpu_graphs = False
    args.reuse_cache = False
    args.bucket_internal = False
    args.disk_offload = False
    args.seed = 27
    args.token = None
    args.assistant_model = None
    args.torch_compile = False
    args.peft_model = None
    args.quant_config = os.getenv("QUANT_CONFIG", "")
    args.bf16 = True
    args.attn_softmax_bf16 = False

    text2cypher_component_name = os.getenv("TEXT2CYPHER_COMPONENT_NAME", "OPEA_TEXT2CYPHER")
    # Initialize OpeaComponentLoader
    loader = OpeaComponentLoader(
        text2cypher_component_name,
        description=f"OPEA TEXT2CYPHER Component: {text2cypher_component_name}",
        config=args.__dict__,
    )

    logger.info("Text2Cypher server started.")
    opea_microservices["opea_service@text2cypher"].start()
