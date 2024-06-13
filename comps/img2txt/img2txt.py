# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import time
from io import BytesIO

import PIL.Image
import requests
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from transformers import pipeline

from comps import Img2TxtDoc, TextDoc, ServiceType, opea_microservices, register_microservice, register_statistics, statistics_dict

def process_image(image, max_len=1344, min_len=672):
    if max(image.size) > max_len:
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
    return image

@register_microservice(
    name="opea_service@img2txt",
    service_type=ServiceType.Img2txt,
    endpoint="/v1/img2txt",
    host="0.0.0.0",
    port=9399,
    input_datatype=Img2TxtDoc,
    output_datatype=TextDoc,
)
@register_statistics(names=["opea_service@img2txt"])
async def img2txt(request: Img2TxtDoc):
    start = time.time()
    img_b64_str = request.image
    prompt = request.prompt
    max_new_tokens = request.max_new_tokens

    print(f"image: {img_b64_str}, prompt: {prompt}, max_new_tokens: {max_new_tokens}")

    # format the prompt
    prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"

    # Decode and Resize the image
    image = PIL.Image.open(BytesIO(base64.b64decode(img_b64_str)))
    image = process_image(image)
    generate_kwargs = {
        "lazy_mode": True,
        "hpu_graphs": True,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": False,
    }
    generator = pipeline(
        "image-to-text",
        model="llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.bfloat16,
        device="hpu",   # Currently only support HPU
    )
    # if use_hpu_graphs:
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    generator.model = wrap_in_hpu_graph(generator.model)
    print("xxxxxxxxxx")
    result = generator([image], prompt=prompt, batch_size=1, generate_kwargs=generate_kwargs)
    print("yyyyyyy")
    result = result[0]["generated_text"].split("ASSISTANT: ")[-1]
    statistics_dict["opea_service@img2txt"].append_latency(time.time() - start, None)
    return TextDoc(text=result)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    # parser.add_argument("--use_hpu_graphs", default=True, action="store_true")
    # parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations for benchmarking.")
    # parser.add_argument("--bf16", default=True, action="store_true")

    # args = parser.parse_args()
    # adapt_transformers_to_gaudi()

    # if args.bf16:
    #     model_dtype = torch.bfloat16
    # else:
    #     model_dtype = torch.float32

    # model_name_or_path = args.model_name_or_path
    # use_hpu_graphs = args.use_hpu_graphs

    # generator = pipeline(
    #     "image-to-text",
    #     model=args.model_name_or_path,
    #     torch_dtype=model_dtype,
    #     device="hpu",   # Currently only support HPU
    # )

    # warmup
    # generate_kwargs = {
    #     "lazy_mode": True,
    #     "hpu_graphs": use_hpu_graphs,
    #     "max_new_tokens": 100,
    #     "ignore_eos": False,
    # }
    # if use_hpu_graphs:
    #     from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        # generator.model = wrap_in_hpu_graph(generator.model)
    # image_paths = ["https://llava-vl.github.io/static/images/view.jpg"]
    # images = []
    # for image_path in image_paths:
    #     images.append(PIL.Image.open(requests.get(image_path, stream=True, timeout=3000).raw))

    # print("[img2txt] img2txt warmup...")
    # for i in range(args.warmup):
    #     generator(
    #         images,
    #         prompt="<image>\nUSER: What's the content of the image?\nASSISTANT:",
    #         batch_size=1,
    #         generate_kwargs=generate_kwargs,
    #     )

    print("[img2txt] img2txt initialized.")
    opea_microservices["opea_service@img2txt"].start()