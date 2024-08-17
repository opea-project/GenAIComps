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
from transformers import pipeline
from comps.embeddings.multimodal_embeddings.bridgetower import BridgeTowerEmbedding

model_name_or_path = None
model_dtype = None
use_hpu_graphs = True

app = FastAPI()

@app.get("/v1/health_check")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model_name_or_path", 
                        type=str, 
                        default="BridgeTower/bridgetower-large-itm-mlm-itc")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    print(f"device: {args.device}")
    if args.device == "hpu":
        try: 
            import habana_frameworks.torch.core as htcore
        except ImportModuleError :  # type: ignore
            print(f"device: hpu is not available. Using cpu instead!")
            args.device = 'cpu'

    model_name_or_path = args.model_name_or_path

    embedding = BridgeTowerEmbedding(device=args.device)

    # warmup
    print("Warmup...")
    image_paths = ["https://llava-vl.github.io/static/images/view.jpg"]    
    example_prompts = ["This is test image!"]
    images = []
    for image_path in image_paths:
        images.append(PIL.Image.open(requests.get(image_path, stream=True, timeout=3000).raw))
    for i in range(args.warmup):
        embedding.embed_image_text_pairs(
            example_prompts,
            images,
            batch_size=1,
        )
    print('Done warmup...')

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
    )