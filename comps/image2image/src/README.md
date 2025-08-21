# Image-to-Image Microservice

The Image-to-Image microservice generates an image based on a provided source image and a descriptive text prompt. This service utilizes a Stable Diffusion (SD) model to perform the image generation task. It takes a source image and text as input and produces a new, modified image as output.

## Table of contents

1.  [Architecture](#architecture)
2.  [Deployment Options](#deployment-options)
3.  [Validated Configurations](#validated-configurations)

## Architecture

The Image-to-Image service is a single microservice that exposes an API endpoint. It receives a request containing a source image URL and a text prompt, processes it using the Stable Diffusion model, and returns the generated image.

- **Image-to-Image Server**: This microservice is the core engine for the image generation task. It can be deployed on both CPU and HPU.

## Deployment Options

For detailed, step-by-step instructions on how to deploy the Image-to-Image microservice using Docker Compose on different Intel platforms, please refer to the deployment guide. The guide contains all necessary steps, including building images, configuring the environment, and running the service.

| Platform          | Deployment Method | Link                                                       |
| ----------------- | ----------------- | ---------------------------------------------------------- |
| Intel Xeon/Gaudi2 | Docker Compose    | [Deployment Guide](../deployment/docker_compose/README.md) |

## Validated Configurations

The following configurations have been validated for the Image-to-Image microservice.

| **Deploy Method** | **Core Models**  | **Platform**      |
| ----------------- | ---------------- | ----------------- |
| Docker Compose    | Stable Diffusion | Intel Xeon/Gaudi2 |
