# ASR Microservice

ASR (Audio-Speech-Recognition) microservice helps users convert speech to text. When building a talking bot with LLM, users will need to convert their audio inputs (What they talk, or Input audio from other sources) to text, so the LLM is able to tokenize the text and generate an answer. This microservice is built for that conversion stage.

## Table of contents

1. [Architecture](#architecture)
2. [Deployment Options](#deployment-options)
3. [Validated Configurations](#validated-configurations)

## Architecture

- **ASR Server**: This microservice is responsible for converting speech audio into text. It receives an audio file as input and returns the transcribed text, enabling downstream applications such as conversational bots to process spoken language. The ASR server supports deployment on both CPU and HPU platforms.
- **Whisper Server**: This microservice is responsible for converting speech audio into text using the Whisper model. It exposes an API endpoint that accepts audio files and returns the transcribed text, supporting both CPU and HPU deployments. The Whisper server acts as the backend for ASR functionality in the overall architecture.
- **FunASR Paraformer Server**: This microservice is responsible for converting speech audio into text using the Paraformer model with the FunASR toolkit. Similar to the Whisper Server, it exposes an API endpoint that accepts audio files and returns the transcribed text, supporting CPU deployments. The FunASR Paraformer server acts as the backend for ASR functionality in the overall architecture.

## Deployment Options

For detailed, step-by-step instructions on how to deploy the ASR microservice using Docker Compose on different Intel platforms, please refer to the deployment guide. The guide contains all necessary steps, including building images, configuring the environment, and running the service.

| Platform          | Deployment Method | Link                                       |
| ----------------- | ----------------- | ------------------------------------------ |
| Intel Xeon/Gaudi2 | Docker Compose    | [Deployment Guide](./README_whisper.md)    |
| Intel Core        | Docker Compose    | [Deployment Guide](./README_paraformer.md) |

## Validated Configurations

The following configurations have been validated for the ASR microservice.

| **Deploy Method** | **Core Models** | **Platform**      |
| ----------------- | --------------- | ----------------- |
| Docker Compose    | Whisper         | Intel Xeon/Gaudi2 |
| Docker Compose    | Paraformer      | Intel Core        |
