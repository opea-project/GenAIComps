# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Stand-alone Stable Video Diffusion FastAPI Server."""

import argparse
import os
import time

import torch
import uvicorn
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    print("SVD generation begin.")
    request_dict = await request.json()
    images_path = request_dict.pop("images_path")

    start = time.time()
    images = [load_image(img) for img in images_path]
    images = [image.resize((1024, 576)) for image in images]

    generator = torch.manual_seed(args.seed)
    frames = pipe(images, decode_chunk_size=8, generator=generator).frames[0]
    video_path = os.path.join(os.getcwd(), args.video_path)
    export_to_video(frames, video_path, fps=7)
    end = time.time()
    print(f"SVD video output in {video_path}, time = {end-start}s")
    return JSONResponse({"video_path": video_path})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9368)
    parser.add_argument("--model_name_or_path", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--video_path", type=str, default="generated.mp4")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    pipe = StableVideoDiffusionPipeline.from_pretrained(args.model_name_or_path)
    print("Stable Video Diffusion model initialized.")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
    )
