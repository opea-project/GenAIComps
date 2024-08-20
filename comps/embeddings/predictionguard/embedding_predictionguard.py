import os
import time
import logging
from fastapi import FastAPI, HTTPException
from predictionguard import PredictionGuard

from comps import (
    EmbedDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

# Initialize Prediction Guard client
client = PredictionGuard()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hard-coded model name
MODEL_NAME = "bridgetower-large-itm-mlm-itc"

@register_microservice(
    name="opea_service@embedding_predictionguard",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
    input_datatype=TextDoc,
    output_datatype=EmbedDoc,
)
@register_statistics(names=["opea_service@embedding_predictionguard"])
def embedding(input: TextDoc) -> EmbedDoc:
    logger.info(f"Received input: {input.text}")

    if not input.text.strip():  # Validate for empty input
        logger.error("Input text is empty. Raising HTTPException with 400 status.")
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    # Check if the model name is supported
    if MODEL_NAME != "bridgetower-large-itm-mlm-itc":
        logger.error(f"Model not supported: {MODEL_NAME}. Raising HTTPException with 500 status.")
        raise HTTPException(status_code=500, detail=f"Model not supported: {MODEL_NAME}")

    start = time.time()
    try:
        response = client.embeddings.create(model=MODEL_NAME, input=[{"text": input.text}])
        logger.info(f"Prediction Guard API response: {response}")

        if "data" not in response or not response["data"]:
            logger.error(f"Failed to generate embeddings for model {MODEL_NAME}.")
            raise HTTPException(status_code=500, detail=f"Failed to generate embeddings with model {MODEL_NAME}")
        
        embed_vector = response["data"][0]["embedding"]
        embed_vector = embed_vector[:512]  # Keep only the first 512 elements
        res = EmbedDoc(text=input.text, embedding=embed_vector)
        statistics_dict["opea_service@embedding_predictionguard"].append_latency(time.time() - start, None)
        return res
    except Exception as e:
        logger.exception("An unexpected error occurred.")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    logger.info(f"Prediction Guard Embedding initialized with model: {MODEL_NAME}")
    opea_microservices["opea_service@embedding_predictionguard"].start()