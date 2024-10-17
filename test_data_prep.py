# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import requests

from comps import CustomLogger

# Initialize custom logger
logger = CustomLogger("docsum_dataprep")
logflag = os.getenv("LOGFLAG", False)

from comps import (
    Base64ByteStrDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    TextDoc,
    Audio2text,
    DocSumDoc
    
)

# Register the microservice
@register_microservice(
    name="opea_service@docsum_dataprep",
    service_type=ServiceType.ASR,
    endpoint="/v1/docsum/dataprep",
    host="0.0.0.0",
    port=7079,
    input_datatype=DocSumDoc,
    output_datatype=Audio2text, # TextDoc, 
)

@register_statistics(names=["opea_service@docsum_dataprep"])
async def audio_to_text(input: DocSumDoc):

    if input.video is not None:
        print("video >>>> ")
        
        inputs = {"byte_str": input.video}
        
        response = requests.post(url=f"{a2t_endpoint}/v1/asr", data=json.dumps(inputs), proxies={"http": None})
        
        # input.audio = 
    
    
    if input.audio is not None:
        inputs = {"audio": input.audio}
        
        print("audio input >>>> ") # , inputs)
        
        # Send the POST request to the ASR endpoint
        response = requests.post(url=f"{a2t_endpoint}/v1/asr", data=json.dumps(inputs), proxies={"http": None})
        response_to_return = response.json()["asr_result"]
        print("audio response >>>> ", response_to_return)
    
    if input.text is not None:
        print("text >>>> ")
        response_to_return = input.text
                
    else:
        print("No input")
            
    return Audio2text(query=response_to_return)
    
    # return Audio2text(query=response.json()["asr_result"]) #.text
    
    # return {"test":"This is a test"}

    # return Audio2text(query={"test":"This is a test"})
    
    
    # try:

        
        
    #     byte_str = input.byte_str
    #     inputs = {"audio": byte_str}
        
    #     if logflag:
    #         logger.info(f"Inputs: {inputs}")

    #     # Send the POST request to the ASR endpoint
    #     response = requests.post(url=f"{a2t_endpoint}/v1/asr", data=json.dumps(inputs), proxies={"http": None})
    #     response.raise_for_status()  # Raise an error for bad status codes
        
    #     if logflag:
    #         logger.info(f"Response: {response.json()}")
        
    #     # Return the transcription result
    #     return Audio2text(query=response.json()["asr_result"]) #.text
    
    
    # except requests.RequestException as e:
    #     logger.error(f"Request to ASR endpoint failed: {e}")
    #     raise
    # except Exception as e:
    #     logger.error(f"An error occurred during audio to text conversion: {e}")
    #     raise

if __name__ == "__main__":
    try:
        # # Get the V2T endpoint from environment variables or use the default
        v2a_endpoint = os.getenv("V2A_ENDPOINT", "http://localhost:7078")
                
        # Get the A2T endpoint from environment variables or use the default
        a2t_endpoint = os.getenv("A2T_ENDPOINT", "http://localhost:7066")
        
        # Log initialization message
        logger.info("[docsum_dataprep - router] docsum_dataprep initialized.")
        
        # Start the microservice
        opea_microservices["opea_service@docsum_dataprep"].start()
        
    except Exception as e:
        logger.error(f"Failed to start the microservice: {e}")
        raise