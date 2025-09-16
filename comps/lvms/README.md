# LVM Microservice

Visual Question and Answering is one of the multimodal tasks empowered by LVMs (Large Visual Models). This microservice supports visual Q&A by using LLaVA as the base large visual model. It accepts two inputs: a prompt and an image. It outputs the answer to the prompt about the image.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Supported Implementations](#supported-implementations)

---

## Overview

Users can configure and deploy LVM-related services based on their specific requirements. This microservice supports a variety of backend implementations, each tailored to different performance, hardware, and model needs, allowing for flexible integration into diverse GenAI workflows.

---

## Key Features

- **Multimodal Interaction**  
  Natively supports question and answering with various visual inputs, including images and videos.

- **Flexible Backends**  
  Integrates with multiple state-of-the-art LVM implementations like LLaVA, LLaMA-Vision, Video-LLaMA, and more.

- **Scalable Deployment**  
  Ready for deployment using Docker, Docker Compose, and Kubernetes, ensuring scalability from local development to production environments.

- **Standardized API**  
  Provides a consistent and simple API endpoint, abstracting the complexities of the different underlying models.

---

## Supported Implementations

The LVM Microservice supports multiple implementation options. Select the one that best fits your use case and follow the linked documentation for detailed setup instructions.

| Implementation           | Description                                                            | Documentation                                           |
| ------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------- |
| **With LLaVA**           | A general-purpose VQA service using the LLaVA model.                   | [README_llava](src/README_llava.md)                     |
| **With TGI LLaVA**       | LLaVA service accelerated by TGI, optimized for Intel Gaudi HPUs.      | [README_llava_tgi](src/README_llava_tgi.md)             |
| **With LLaMA-Vision**    | VQA service leveraging the LLaMA-Vision model.                         | [README_llama_vision](src/README_llama_vision.md)       |
| **With Video-LLaMA**     | A specialized service for performing VQA on video inputs.              | [README_video_llama](src/README_video_llama.md)         |
| **With vLLM**            | High-throughput LVM serving accelerated by vLLM on Intel Gaudi HPUs.   | [README_vllm](src/README_vllm.md)                       |
| **With vLLM-IPEX**       | High-throughput LVM serving accelerated by vLLM-IPEX on Intel Arc GPUs | [README_vllm_ipex](src/README_vllm_ipex.md)             |
| **With PredictionGuard** | LVM service using Prediction Guard with built-in safety features.      | [README_predictionguard](src/README_predictionguard.md) |
