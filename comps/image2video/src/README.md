# Image-to-Video Microservice

The Image-to-Video microservice generates a video based on a provided source image. This service utilizes the Stable Video Diffusion (SVD) model to perform the video generation task. It takes a source image as input and produces a short video clip as output.

## Table of contents

1.  [Architecture](#architecture)
2.  [Deployment Options](#deployment-options)
3.  [Validated Configurations](#validated-configurations)

## Architecture

The Image-to-Video service is a single microservice that exposes an API endpoint. It receives a request containing a source image, processes it using the Stable Video Diffusion model, and returns the generated video.

- **Image-to-Video Server**: This microservice is the core engine for the video generation task. It can be deployed on both CPU and HPU.

## Deployment Options

For detailed, step-by-step instructions on how to deploy the Image-to-Video microservice using Docker Compose on different Intel platforms, please refer to the deployment guide. The guide contains all necessary steps, including building images, configuring the environment, and running the service.

| Platform          | Deployment Method | Link                                                       |
| ----------------- | ----------------- | ---------------------------------------------------------- |
| Intel Xeon/Gaudi2 | Docker Compose    | [Deployment Guide](../deployment/docker_compose/README.md) |

## Validated Configurations

The following configurations have been validated for the Image-to-Video microservice.

| **Deploy Method** | **Core Models**              | **Platform**      |
| ----------------- | ---------------------------- | ----------------- |
| Docker Compose    | Stable Video Diffusion (SVD) | Intel Xeon/Gaudi2 |
