# Avatar Animation Microservice

The avatar animation model is a combination of two models: Wav2Lip and GAN-based face generator (GFPGAN). The Wav2Lip model is used to generate lip movements from an audio file, and the GFPGAN model is used to generate a high-quality face image from a low-quality face image. The avatar animation microservices takes an audio piece and a low-quality face image/video as input, fuses mel-spectrogram from the audio with frame(s) from the image/video, and generates a high-quality video of the face image with lip movements synchronized with the audio.

## Table of contents

1. [Architecture](#architecture)
2. [Deployment Options](#deployment-options)
3. [Validated Configurations](#validated-configurations)

## Architecture

The Avatar Animation service consists of two primary microservices:

- **Wav2Lip Server**: This microservice is the core engine for lip synchronization. It takes an audio file and a face image/video as input and generates a video where the lip movements match the provided audio. It can be deployed on both CPU and HPU.
- **Animation Server**: This microservice acts as an orchestrator or gateway. It exposes a single endpoint for the user, receives the request, forwards it to the Wav2Lip server for processing, and then returns the final generated video to the user.

## Deployment Options

For detailed, step-by-step instructions on how to deploy the Avatar Animation microservice using Docker Compose on different Intel platforms, please refer to the deployment guide. The guide contains all necessary steps, including building images, configuring the environment, and running the service.

| Platform          | Deployment Method | Link                                                       |
| ----------------- | ----------------- | ---------------------------------------------------------- |
| Intel Xeon/Gaudi2 | Docker Compose    | [Deployment Guide](../deployment/docker_compose/README.md) |

## Validated Configurations

The following configurations have been validated for the Avatar Animation microservice.

| **Deploy Method** | **Core Models** | **Platform**      |
| ----------------- | --------------- | ----------------- |
| Docker Compose    | Wav2Lip, GFPGAN | Intel Xeon/Gaudi2 |
