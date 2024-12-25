# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import os
import threading
import time

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

from comps import (
    CustomLogger,
    SDImg2ImgInputs,
    SDOutputs,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from io import BytesIO
from PIL import Image

logger = CustomLogger("image2image_sdwebui")
pipe = None
args = None
initialization_lock = threading.Lock()
initialized = False


def initialize():
    global pipe, args, initialized
    with initialization_lock:
        if not initialized:
            # initialize model and tokenizer
            if os.getenv("MODEL", None):
                args.model_name_or_path = os.getenv("MODEL")
            kwargs = {}
            if args.bf16:
                kwargs["torch_dtype"] = torch.bfloat16
            if not args.token:
                args.token = os.getenv("HF_TOKEN")
            if args.device == "hpu":
                kwargs.update(
                    {
                        "use_habana": True,
                        "use_hpu_graphs": args.use_hpu_graphs,
                        "gaudi_config": "Habana/stable-diffusion",
                        "token": args.token,
                    }
                )
                if "stable-diffusion-xl" in args.model_name_or_path:
                    from optimum.habana.diffusers import GaudiStableDiffusionXLImg2ImgPipeline

                    pipe = GaudiStableDiffusionXLImg2ImgPipeline.from_pretrained(
                        args.model_name_or_path,
                        **kwargs,
                    )
                elif "stable-diffusion" in args.model_name_or_path:
                    from optimum.habana.diffusers import GaudiStableDiffusionImg2ImgPipeline

                    pipe = GaudiStableDiffusionImg2ImgPipeline.from_pretrained(
                        args.model_name_or_path,
                        **kwargs,
                    )
                else:
                    raise NotImplementedError(
                        "Only support stable-diffusion-xl now, " + f"model {args.model_name_or_path} not supported."
                    )
            elif args.device == "cpu":
                pipe = AutoPipelineForImage2Image.from_pretrained(args.model_name_or_path, token=args.token, **kwargs)
            else:
                raise NotImplementedError(f"Only support cpu and hpu device now, device {args.device} not supported.")
            logger.info("Stable Diffusion model initialized.")
            initialized = True


@register_microservice(
    name="opea_service@image2image_sdwebui",
    service_type=ServiceType.IMAGE2IMAGE,
    endpoint="/sdapi/v1/img2img",
    host="0.0.0.0",
    port=9389,
    input_datatype=SDImg2ImgInputs,
    output_datatype=SDOutputs,
)
@register_statistics(names=["opea_service@image2image_sdwebui"])
def image2image(input: SDImg2ImgInputs):
    initialize()
    start = time.time()
    image_byte = base64.b64decode(input.image)
    image_io = BytesIO(image_byte)
    image = Image.open(image_io)
    image = load_image(image).convert("RGB")
    prompt = input.prompt
    num_inference_steps = input.num_inference_steps
    guidance_scale = input.guidance_scale
    num_images_per_prompt = input.num_images_per_prompt
    seed = input.seed
    negative_prompt = input.negative_prompt
    height = input.height
    width = input.width
    strength = input.strength

    images = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
    ).images

    generator = torch.manual_seed(seed)
    images = pipe(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
    ).images
    image_path = os.path.join(os.getcwd(), prompt.strip().replace(" ", "_").replace("/", ""))
    os.makedirs(image_path, exist_ok=True)
    results = []
    for i, image in enumerate(images):
        save_path = os.path.join(image_path, f"image_{i+1}.png")
        image.save(save_path)
        with open(save_path, "rb") as f:
            bytes = f.read()
        b64_str = base64.b64encode(bytes).decode()
        results.append(b64_str)
    statistics_dict["opea_service@image2image_sdwebui"].append_latency(time.time() - start, None)
    return SDOutputs(images=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--use_hpu_graphs", default=False, action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    logger.info("Image2image server started.")
    opea_microservices["opea_service@image2image_sdwebui"].start()
