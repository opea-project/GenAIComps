# Embeddings Microservice

The Embedding Microservice is designed to efficiently convert textual strings into vectorized embeddings, facilitating seamless integration into various machine learning and data processing workflows. This service utilizes advanced algorithms to generate high-quality embeddings that capture the semantic essence of the input text, making it ideal for applications in natural language processing, information retrieval, and similar fields.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Supported Implementations](#supported-implementations)

---

## Overview

Users are able to configure and build embedding-related services according to their actual needs. The microservice supports different backend implementations to suit various performance, deployment, and model requirements.

---

## Key Features

- **High Performance**  
  Optimized for quick and reliable conversion of textual data into vector embeddings.

- **Scalability**  
  Built to handle high volumes of requests simultaneously, ensuring robust performance even under heavy loads.

- **Ease of Integration**  
  Provides a simple and intuitive API, allowing for straightforward integration into existing systems and workflows.

- **Customizable**  
  Supports configuration and customization to meet specific use case requirements, including different embedding models and preprocessing techniques.

---

## Supported Implementations

The Embeddings Microservice supports multiple implementation options to suit different deployment and usage scenarios. Each implementation includes its own configuration and setup instructions:

| Implementation                  | Description                                                     | Documentation                                           |
| ------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------- |
| **With OVMS**                   | Embedding microservice using OpenVINO Model Server (OVMS)       | [README_OVMS](src/README_ovms.md)                       |
| **With TEI**                    | TEI-based embedding microservice for efficient text processing  | [README_TEI](src/README_tei.md)                         |
| **With Prediction Guard**       | Embedding service using Prediction Guard with safety filters    | [README_PredictionGuard](src/README_predictionguard.md) |
| **With Multimodal CLIP**        | Multimodal embedding service using CLIP for text and image data | [README_CLIP](src/README_clip.md)                       |
| **With Multimodal BridgeTower** | Multimodal embedding service using BridgeTower                  | [README_BridgeTower](src/README_bridgetower.md)         |
