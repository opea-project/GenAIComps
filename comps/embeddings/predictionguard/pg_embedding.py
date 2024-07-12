import os
import json
import time
from predictionguard import PredictionGuard
from flask import Flask, request, jsonify

from comps import (
    EmbedDoc768,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

app = Flask(__name__)

# Set your Prediction Guard API key
PREDICTIONGUARD_API_KEY = os.getenv("PREDICTIONGUARD_API_KEY", "<your_api_key>")

client = PredictionGuard(api_key=PREDICTIONGUARD_API_KEY)

@register_microservice(
    name="opea_service@predictionguard_embedding",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
    input_datatype=TextDoc,
    output_datatype=EmbedDoc768,
)
@traceable(run_type="embedding")
@register_statistics(names=["opea_service@predictionguard_embedding"])
def embedding(input: TextDoc) -> EmbedDoc768:
    start = time.time()
    response = client.embeddings.create(
        model="bridgetower-large-itm-mlm-itc",
        input=[{"text": input.text}]
    )
    embed_vector = response["data"][0]["embedding"]
    embed_vector = embed_vector[:768]  # Keep only the first 768 elements
    res = EmbedDoc768(text=input.text, embedding=embed_vector)
    statistics_dict["opea_service@predictionguard_embedding"].append_latency(time.time() - start, None)
    return res

@app.route('/v1/health_check', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    print("Prediction Guard Embedding initialized.")
    opea_microservices["opea_service@predictionguard_embedding"].start()