# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Stand-alone Stable Diffusion FastAPI Server."""

import argparse
import os
import time

import torch
import uvicorn
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    print("SD generation begin.")
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    num_images_per_prompt = request_dict.pop("num_images_per_prompt", 1)

    start = time.time()
    generator = torch.manual_seed(args.seed)
    images = pipe(prompt, generator=generator, num_images_per_prompt=num_images_per_prompt).images
    image_path = os.path.join(os.getcwd(), prompt.strip().replace(" ", "_").replace("/", ""))
    os.makedirs(image_path, exist_ok=True)
    for i, image in enumerate(images):
        image.save(os.path.join(image_path, f"image_{i+1}.png"))
    end = time.time()
    print(f"SD Images output in {image_path}, time = {end-start}s")
    return JSONResponse({"image_path": image_path})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9378)
    parser.add_argument("--model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--use_hpu_graphs", default=False, action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()
    if not args.token:
        args.token = os.getenv("HF_TOKEN")
    if args.device == "hpu":
        kwargs = {
            "use_habana": True,
            "use_hpu_graphs": args.use_hpu_graphs,
            "gaudi_config": "Habana/stable-diffusion",
            "token": args.token
        }
        if args.bf16:
            kwargs["torch_dtype"] = torch.bfloat16
        if "stable-diffusion-3" in args.model_name_or_path:
            from optimum.habana.diffusers import GaudiStableDiffusion3Pipeline

            pipe = GaudiStableDiffusion3Pipeline.from_pretrained(
                args.model_name_or_path,
                **kwargs,
            )
        elif "stable-diffusion-xl" in args.model_name_or_path:
            from optimum.habana.diffusers import GaudiStableDiffusionXLPipeline

            pipe = GaudiStableDiffusionXLPipeline.from_pretrained(
                args.model_name_or_path,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "Only support stable-diffusion-3 and stable-diffusion-xl now, " + \
                f"model {args.model_name_or_path} not supported."
            )
    elif args.device == "cpu":
        pipe = DiffusionPipeline.from_pretrained(args.model_name_or_path, token=args.token)
    else:
        raise NotImplementedError(f"Only support cpu and hpu device now, device {args.device} not supported.")
    print("Stable Diffusion model initialized.")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
    )
