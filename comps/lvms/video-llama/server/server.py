# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Stand-alone video llama FastAPI Server."""

import argparse
from threading import Thread
import yaml

import decord
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from transformers import set_seed, TextIteratorStreamer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from extract_vl_embedding import VLEmbeddingExtractor as VL
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat

decord.bridge.set_bridge('torch')
set_seed(22)

context_db = None
streamer = None
chat = None

app = FastAPI()

class videoInfo(BaseModel):
  video_path: str = Field(..., description="Supported: chroma, vdms, hsm" )
  start_time: float = Field(..., descrciption="video clip start time in seconds", example=0.0 )
  duration: float = Field(..., description="video clip duration in seconds", example=10.0)

class GenerateRequest(BaseModel):
  video: videoInfo
  prompt: str = Field(..., description="Query for Video-LLama", example="What is the man doing?")
  max_new_tokens: int = Field(default=512, description="Maximum number of tokens to generate", example=512) # 

def construct_instructions():
    instructions = [
        """ Identify the person [with specific features / seen at a specific location / performing a specific action] in the provided data based on the video content. 
        Describe in detail the relevant actions of the individuals mentioned in the question. 
        Provide full details of their actions being performed and roles. Focus on the individual and the actions being performed.
        Exclude information about their age and items on the shelf that are not directly observable. 
        Do not mention items on the shelf that are not  visible. \
        Exclude information about the background and surrounding details.
        Ensure all information is distinct, accurate, and directly observable. 
        Do not repeat actions of individuals and do not mention anything about other persons not visible in the video.
        Mention actions and roles once only.
        """,
        
        """Analyze the provided data to recognize and describe the activities performed by individuals.
        Specify the type of activity and any relevant contextual details, 
        Do not give repetitions, always give distinct and accurate information only.""",
        
        """Determine the interactions between individuals and items in the provided data. 
        Describe the nature of the interaction between individuals and the items involved. 
        Provide full details of their relevant actions and roles. Focus on the individuals and the action being performed by them.
        Exclude information about their age and items on the shelf that are not directly observable. 
        Exclude information about the background and surrounding details.
        Ensure all information is distinct, accurate, and directly observable. 
        Do not repeat actions of individuals and do not mention anything about other persons not visible in the video.
        Do not mention  items on the shelf that are not observable. \
        """,
        
        """Analyze the provided data to answer queries based on specific time intervals.
        Provide detailed information corresponding to the specified time frames,
        Do not give repetitions, always give distinct and accurate information only.""",
        
        """Identify individuals based on their appearance as described in the provided data.
        Provide details about their identity and actions,
        Do not give repetitions, always give distinct and accurate information only.""",
        
        """Answer questions related to events and activities that occurred on a specific day.
        Provide a detailed account of the events,
        Do not give repetitions, always give distinct and accurate information only."""
    ]
    HFembeddings = HuggingFaceEmbeddings(model_kwargs = {'device': 'cpu'})
    context = FAISS.from_texts(instructions, HFembeddings)
    return context

def read_config(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return data

def get_context(query, context):
    context = context.similarity_search(query)
    return [i.page_content for i in context]

def chat_reset(chat_state, img_list):
    print("-"*30)
    print("resetting chatState")
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list


def inference(chat, streamer, video: videoInfo, instruction: str):
    video_path = video.video_path
    start_time = video.start_time
    duration = video.duration
    
    chat.upload_video_without_audio(video_path, start_time, duration)
    chat.ask("<rag_prompt>"+instruction)#, chat_state) # the state is reserved.
    chat.answer(max_new_tokens=150, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=0.02, max_length=2000, keep_conv_hist=True, streamer=streamer)

def stream_res(video, instruction):
    thread = Thread(target=inference, args=(chat, streamer, video, instruction))  # Pass streamer to inference
    thread.start()
    for text in streamer:
        yield text



@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate", response_class=StreamingResponse)
async def generate(request: GenerateRequest) -> StreamingResponse:
    print("Video-Llama generation begin.")
    
    # format context and instruction
    instruction = f"{get_context(request.prompt,context_db)[0]}: {request.prompt}"
    print("instruction:",instruction)
    
    return StreamingResponse(stream_res(request.video, instruction))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--cfg_path", type=str, default="video_llama_config/video_llama_eval_only_vl.yaml")
    parser.add_argument("--model_type", type=str, default="llama_v2")
    # parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-1.5-7b-hf")


    args = parser.parse_args()
        
    # format context and instruction
    context_db = construct_instructions()
    
    # create chat
    video_llama = VL(cfg_path=args.cfg_path, model_type=args.model_type)
    tokenizer = video_llama.model.llama_tokenizer
    
    # global streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    
    vis_processor_cfg = video_llama.cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    # global chat
    chat = Chat(video_llama.model, vis_processor, device="cpu")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug"
        )
