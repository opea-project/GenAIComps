import json
import os
# import time

# from fastapi import FastAPI, Request
# import numpy as np
import requests

from comps import (
    Base64ByteStrDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    CustomLogger
)

# Initialize custom logger
logger = CustomLogger("video2audio")
logflag = os.getenv("LOGFLAG", False)

# Register the microservice
@register_microservice(
    name="opea_service@video2audio",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/video2audio",
    host="0.0.0.0",
    port=7078,
    input_datatype=Base64ByteStrDoc,
    output_datatype=Base64ByteStrDoc,
)
@register_statistics(names=["opea_service@video2audio"])
async def audio_to_text(request: Base64ByteStrDoc):
    """
    Convert video to audio and return the result in base64 format.
    
    Args:
        request (Base64ByteStrDoc): The incoming request containing the video in base64 format.
    
    Returns:
        Base64ByteStrDoc: The response containing the audio in base64 format.
    """
    
    try:
        logger.info("Received request for video to audio conversion.")
        
        byte_str = request.byte_str
        inputs = {"video": byte_str}
        
        logger.debug(f"Sending request to video-to-audio endpoint: {v2a_endpoint}/v1/v2asr")
        response = requests.post(url=f"{v2a_endpoint}/v1/v2asr", data=json.dumps(inputs), proxies={"http": None})
        
        response.raise_for_status()  # Raise an error for bad status codes
        
        logger.info("Successfully converted video to audio.")
        return Base64ByteStrDoc(byte_str=response.json()['v2a_result'])
    
    except requests.RequestException as e:
        logger.error(f"Request to video-to-audio endpoint failed: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during video to audio conversion: {e}")
        raise

if __name__ == "__main__":
    try:
        # Get the video-to-audio endpoint from environment variables or use the default
        v2a_endpoint = os.getenv("VIDEO2AUDIO_ENDPOINT", "http://localhost:7077")
        # v2a_endpoint = os.getenv("VIDEO2AUDIO_ENDPOINT", "dataprep-docsum-video2audio:7077")
        
        # Log initialization message
        logger.info("[video2audio - router] VIDEO2AUDIO initialized.")
        
        # Start the microservice
        opea_microservices["opea_service@video2audio"].start()
        
    except Exception as e:
        logger.error(f"Failed to start the microservice: {e}")
        raise

