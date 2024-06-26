# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

# __serve_example_begin__
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 5,
    },
    max_ongoing_requests=10,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Determine the name of the served model for the OpenAI client.
        if engine_args.served_model_name is not None:
            served_model_names = engine_args.served_model_name
        else:
            served_model_names = [engine_args.model]
        self.openai_serving_chat = OpenAIServingChat(
            self.engine, served_model_names, response_role, lora_modules, chat_template
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    parser = make_arg_parser()
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    if "device" in cli_args.keys():
        device = cli_args.pop("device")
    else:
        try:
            from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

            initialize_distributed_hpu()
            torch.zeros(1).to("hpu")
            device = "HPU"
        except Exception:
            device = "GPU"
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp):
        pg_resources.append({"CPU": 1, device: 1})  # for the vLLM actors

    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    return VLLMDeployment.options(placement_group_bundles=pg_resources, placement_group_strategy="STRICT_PACK").bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
    )


# __serve_example_end__


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Serve vLLM models with Ray Serve.", add_help=True)
    parser.add_argument("--port_number", default=8008, type=int, help="Port number to serve on.")
    parser.add_argument(
        "--model_id_or_path", default="NousResearch/Meta-Llama-3-8B-Instruct", type=str, help="Model id or path."
    )
    parser.add_argument(
        "--chat_processor", default="ChatModelNoFormat", type=str, help="Chat processor for aligning the prompts."
    )
    parser.add_argument("--tensor_parallel_size", default=1, type=int, help="parallel nodes number for 'hpu' mode.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(argv)

    serve.run(
        build_app({"model": args.model_id_or_path, "tensor-parallel-size": args.tensor_parallel_size, "device": "HPU"})
    )
    # __query_example_begin__
    # from openai import OpenAI

    # # Note: Ray Serve doesn't support all OpenAI client arguments and may ignore some.
    # client = OpenAI(
    #     # Replace the URL if deploying your app remotely
    #     # (e.g., on Anyscale or KubeRay).
    #     base_url="http://localhost:8000/v1",
    #     api_key="NOT A REAL KEY",
    # )
    # chat_completion = client.chat.completions.create(
    #     model="NousResearch/Meta-Llama-3-8B-Instruct",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {
    #             "role": "user",
    #             "content": "What are some highly rated restaurants in San Francisco?'",
    #         },
    #     ],
    #     temperature=0.01,
    #     stream=True,
    # )

    # for chat in chat_completion:
    #     if chat.choices[0].delta.content is not None:
    #         print(chat.choices[0].delta.content, end="")
    # __query_example_end__


if __name__ == "__main__":
    main(sys.argv[1:])
