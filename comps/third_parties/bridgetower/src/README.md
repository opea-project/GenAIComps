# Multimodal Embeddings Microservice with BridgeTower

The Multimodal Embedding Microservice is designed to efficiently convert pairs of textual string and image into vectorized embeddings, facilitating seamless integration into various machine learning and data processing workflows. This service utilizes advanced algorithms to generate high-quality embeddings that capture the joint semantic essence of the input text-and-image pairs, making it ideal for applications in multi-modal data processing, information retrieval, and similar fields.

Key Features:

**High Performance**: Optimized for quick and reliable conversion of textual data and image inputs into vector embeddings.

**Scalability**: Built to handle high volumes of requests simultaneously, ensuring robust performance even under heavy loads.

**Ease of Integration**: Provides a simple and intuitive API, allowing for straightforward integration into existing systems and workflows.

**Customizable**: Supports configuration and customization to meet specific use case requirements, including different embedding models and preprocessing techniques.

Users are albe to configure and build embedding-related services according to their actual needs.

Currently, we employ [**BridgeTower**](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-gaudi) model for MMEI and provide two ways to start MMEI:

## 🚀1. Start MMEI on Gaudi2 HPU

- Gaudi2 HPU

```bash
# Define port to use for the embeddings microservice
export EMBEDDER_PORT=8080

cd ../../../../../../../
docker build -t opea/embedding-multimodal-bridgetower-gaudi:latest --build-arg EMBEDDER_PORT=$EMBEDDER_PORT --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/bridgetower/src/Dockerfile.intel_hpu .
cd comps/third_parties/bridgetower/deployment/docker_compose/
docker compose -f compose.yaml up -d multimodal-bridgetower-embedding-gaudi-serving
```

## 🚀2. Start MMEI on Xeon CPU

- Xeon CPU

```bash
# Define port to use for the embeddings microservice
export EMBEDDER_PORT=8080

cd ../../../../../../../
docker build -t opea/embedding-multimodal-bridgetower:latest --build-arg EMBEDDER_PORT=$EMBEDDER_PORT --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/bridgetower/src/Dockerfile .
cd comps/third_parties/bridgetower/deployment/docker_compose/
docker compose -f compose.yaml up -d multimodal-bridgetower-embedding-serving
```

## 🚀3. Access the service

Then you need to test your MMEI service using the following commands:

```bash
curl http://localhost:$EMBEDDER_PORT/v1/encode \
     -X POST \
     -H "Content-Type:application/json" \
     -d '{"text":"This is example"}'
```

To compute a joint embedding of an image-text pair, a base64 encoded image can be passed along with text:

```bash
curl -X POST http://localhost:$EMBEDDER_PORT/v1/encode \
-H "Content-Type: application/json" \
-d '{"text":  "This is some sample text.", "img_b64_str" : "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC"}'
```
