import argparse
import os

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, JSONResponse
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/v1/health")
async def health() -> JSONResponse:
    """Health check."""
    print("***** in s1 *****")
    # return Response(status_code=200)
    return {"data": "***** in s1 *****"}

@app.post("/v1/add")
async def add(request: Request) -> JSONResponse:
    req = await request.json()
    number = req["number"]
    number += 1
    return {"number": number}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)