
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
from tqdm import tqdm
from comps import opea_microservices, register_microservice
from utils.utils import read_config, process_all_videos
from utils import store_embeddings
from utils.vclip import vCLIP
from fastapi import File, HTTPException, UploadFile
import uuid
from typing import Any, Dict, Iterable, List, Optional, Type, Union
import shutil

VECTORDB_SERVICE_HOST_IP = os.getenv("VDMS_HOST", "0.0.0.0")

def setup_vclip_model(config, device="cpu"):
    model = vCLIP(config)
    return model

def read_json(path):
    with open(path) as f:
        x = json.load(f)
    return x

def store_into_vectordb(vs, metadata_file_path, embedding_model, config):
    GMetadata = read_json(metadata_file_path)
    global_counter = 0

    total_videos = len(GMetadata.keys())
    
    for idx, (video, data) in enumerate(tqdm(GMetadata.items())):
        image_name_list = []
        embedding_list = []
        metadata_list = []
        ids = []
        
        if config['embeddings']['type'] == 'video':
            data['video'] = video
            video_name_list = [data["video_path"]]
            metadata_list = [data]
            if vs.selected_db == 'vdms':
                vs.video_db.add_videos(
                    paths=video_name_list,
                    metadatas=metadata_list,
                    start_time=[data['timestamp']],
                    clip_duration=[data['clip_duration']]
                )
            else:
                print(f"ERROR: selected_db {vs.selected_db} not supported. Supported:[vdms]")

    # clean up tmp_ folders containing frames (jpeg)
    for i in os.listdir():
        if i.startswith("tmp_"):
            print("removing tmp_*")
            os.system(f"rm -r tmp_*")
            print("done.")
            break
            
def generate_video_id():
    """Generates a unique identifier for a video file."""
    return str(uuid.uuid4())        

def generate_embeddings(config, embedding_model, vs):
    print('inside generate')
    process_all_videos(config)
    global_metadata_file_path = os.path.join(config["meta_output_dir"], 'metadata.json')
    print(f'global metadata file available at {global_metadata_file_path}')
    store_into_vectordb(vs, global_metadata_file_path, embedding_model, config)
      
@register_microservice(
    name="opea_service@prepare_doc_vdms",
    endpoint="/v1/dataprep",
    host="0.0.0.0",
    port=6007
)

def process_videos(files: List[UploadFile] = File(None)):
    """Ingest videos to VDMS."""
    
    config= config = read_config('./config.yaml')
    meanclip_cfg = {"model_name": config['embeddings']['vclip_model_name'], "num_frm": config['embeddings']['vclip_num_frm']}
    generate_frames = config['generate_frames']
    path = config['videos']
    meta_output_dir = config['meta_output_dir']
    emb_path = config['embeddings']['path']
    host = VECTORDB_SERVICE_HOST_IP
    port = int(config['vector_db']['port'])
    selected_db = config['vector_db']['choice_of_db']
    print(f"Parsing videos {path}.")
    
    #Saving videos
    if files:
        video_files = []
        for file in files:
            if os.path.splitext(file.filename)[1] == ".mp4":
                video_files.append(file)
            else:
                raise HTTPException(
                    status_code=400, detail=f"File {file.filename} is not an mp4 file. Please upload mp4 files only."
                )

        for video_file in video_files:
            video_id = generate_video_id()
            video_name = os.path.splitext(video_file.filename)[0]
            video_file_name = f"{video_name}_{video_id}.mp4"
            video_dir_name = os.path.splitext(video_file_name)[0]
            # Save video file in upload_directory
            with open(os.path.join(path, video_file_name), "wb") as f:
                shutil.copyfileobj(video_file.file, f)

    
    # Creating DB
    print ('Creating DB with video embedding and metadata support, \nIt may take few minutes to download and load all required models if you are running for first time.')
    print('Connecting to {} at {}:{}'.format(selected_db, host, port))
    #check embedding type
    if 'video' == 'video':
        # init meanclip model
        model = setup_vclip_model(meanclip_cfg, device="cpu")
        print('init model')
        vs = store_embeddings.VideoVS(host, port, selected_db, model)
        print('init vector store')
    else:
        print(f"ERROR: Selected embedding type in config.yaml {config['embeddings']['type']} is not in [\'video\', \'frame\']")
        return
    generate_embeddings(config, model, vs)
    print('done............success..............')


if __name__ == "__main__":
    opea_microservices["opea_service@prepare_doc_vdms"].start()