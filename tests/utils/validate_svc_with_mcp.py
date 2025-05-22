#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import os
import sys
# from random import random
import random
import requests
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))


async def validate_svc(ip_address, service_port, service_type):

    endpoint = f"http://{ip_address}:{service_port}"

    async with sse_client(endpoint + "/sse") as streams:
        async with ClientSession(*streams) as session:
            result = await session.initialize()
            if service_type == "asr":
                url = "https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav"
                response = requests.get(url)
                response.raise_for_status()  # Ensure the download succeeded
                binary_data = response.content
                base64_str = base64.b64encode(binary_data).decode("utf-8")
                input_dict = {"file": base64_str, "model": "openai/whisper-small", "language": "english"}
                tool_result = await session.call_tool(
                    "audio_to_text",
                    input_dict,
                )
                result_content = tool_result.content
                # Check result
                if json.loads(result_content[0].text)["text"].startswith("who is"):
                    print("Result correct.")
                else:
                    print(f"Result wrong. Received was {result_content}")
                    exit(1)
            elif service_type == "tts":
                input_dict = {"request": {"input": "Hi there, welcome to OPEA."}}
                tool_result = await session.call_tool(
                    "text_to_speech",
                    input_dict,
                )
                result_content = tool_result.content
                # Check result
                audio_str = json.loads(result_content[0].text).get("audio_str", "")
                if audio_str.startswith("Ukl"):  # "Ukl" indicates likely WAV header
                    audio_data = base64.b64decode(audio_str)
                    with open("output.wav", "wb") as f:
                        f.write(audio_data)
                    with open("output.wav", "rb") as f:
                        header = f.read(4)
                    if header == b"RIFF":
                        print("Result correct.")
                    else:
                        print(f"Invalid WAV file: starts with {header}")
                else:
                    print(f"Result wrong. Received was {result_content}")
                    exit(1)
            elif service_type == "animation":
                with open(os.path.join(root_dir, "comps/animation/src/assets/audio/sample_question.json"), "r") as file:
                    input_dict = json.load(file)
                tool_result = await session.call_tool(
                    "animate",
                    {"audio":input_dict},
                )
                result_content = tool_result.content
                animate_result = json.loads(result_content[0].text).get("video_path", [])
                if bool(animate_result):
                    print("Result correct.")
                else:
                    print(f"Result wrong. Received was {tool_result.content}")
                    exit(1)
            elif service_type == "image2image":
                input_dict = {"image": "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png", "prompt":"a photo of an astronaut riding a horse on mars", "num_images_per_prompt":1}
                tool_result = await session.call_tool(
                    "image2image",
                    {"input": input_dict},
                )
                result_content = tool_result.content
                svc_result = json.loads(result_content[0].text).get("images", [])

                if bool(svc_result):
                    print("Result correct.")
                else:
                    print(f"Result wrong. Received was {tool_result.content}")
                    exit(1)
            elif service_type == "retriever":
                dummy_embedding = [random.uniform(-1, 1) for _ in range(768)]
                print(dummy_embedding)
                input_dict = {"input": {"text": "What is OPEA?", "embedding": dummy_embedding}}
                tool_result = await session.call_tool(
                    "retrieve_docs",
                    input_dict,
                )
                result_content = tool_result.content
                print(result_content)
                retrieved_docs = json.loads(result_content[0].text).get("retrieved_docs", [])
                if len(retrieved_docs) >= 1 and any(
                        "OPEA" in retrieved_docs[i]["text"] for i in range(len(retrieved_docs))
                ):
                    print("Result correct.")
                else:
                    print(f"Result wrong. Received was {tool_result.content}")
            elif service_type == "retriever":
                dummy_embedding = [random.uniform(-1, 1) for _ in range(768)]
                input_dict = {"input": {"text": "What is OPEA?", "embedding": dummy_embedding}}
                tool_result = await session.call_tool(
                    "retrieve_docs",
                    input_dict,
                )
                result_content = tool_result.content
                retrieved_docs = json.loads(result_content[0].text).get("retrieved_docs", [])
                if len(retrieved_docs) >= 1 and any(
                        "OPEA" in retrieved_docs[i]["text"] for i in range(len(retrieved_docs))
                ):
                    print("Result correct.")
                else:
                    print(f"Result wrong. Received was {tool_result.content}")

            else:
                print(f"Unknown service type: {service_type}")
                exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 validate_svc_with_mcp.py <ip_address> <service_port> <service_type>")
        exit(1)
    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    service_type = sys.argv[3]
    asyncio.run(validate_svc(ip_address, service_port, service_type))
