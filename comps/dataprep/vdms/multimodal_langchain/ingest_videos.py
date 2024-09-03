
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
from tqdm import tqdm
from comps import DocPath, opea_microservices, opea_telemetry, register_microservice
from utils.utils import read_config, process_all_videos
from utils import store_embeddings
from utils.vclip import vCLIP


VECTORDB_SERVICE_HOST_IP = os.getenv("VECTORDB_SERVICE_HOST_IP", "0.0.0.0")

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
    port=6007,
    input_datatype=DocPath,
    output_datatype=None,
)
@opea_telemetry
def process_videos(doc_path: DocPath):
    """Ingest videos to VDMS."""
    path = doc_path.path
    print(f"Parsing videos {path}.")

    #################
    #set config_file
    #################
    
    config= config = read_config('./config.yaml')
    meanclip_cfg = {"model_name": config['embeddings']['vclip_model_name'], "num_frm": config['embeddings']['vclip_num_frm']}
    generate_frames = config['generate_frames']
    path = config['videos']
    meta_output_dir = config['meta_output_dir']
    emb_path = config['embeddings']['path']
    host = VECTORDB_SERVICE_HOST_IP
    port = int(config['vector_db']['port'])
    selected_db = config['vector_db']['choice_of_db']
    
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