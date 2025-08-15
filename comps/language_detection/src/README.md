# Language Detection microservice

The Language Detection microservice can be run in 2 modes:

1. Pipeline: This mode adds multilingual support to ChatQnA pipelines. The microservice detects the language of the user's query as well as the LLM generated response to set up a prompt for translation.

2. Standalone: This mode supports standalone translation. The microservice detects the language of the provided text. It then sets up a prompt for translating the provided text from the source language (detected language) to the provided target language.

## Table of contents

1. [Architecture](#architecture)
2. [Deployment Options](#deployment-options)
3. [Validated Configurations](#validated-configurations)

## Architecture

The language-detection service consists of a primary microservices:

- **language-detection**: This microservice detects the language of input text. It is used to identify the source and target languages for translation tasks in both pipeline and standalone modes. The service can be deployed on CPU and HPU platforms.

## Deployment Options

For detailed, step-by-step instructions on how to deploy the language-detection microservice using Docker Compose on different Intel platforms, please refer to the deployment guide. The guide contains all necessary steps, including building images, configuring the environment, and running the service.

| Platform   | Deployment Method | Link                                                       |
| ---------- | ----------------- | ---------------------------------------------------------- |
| Intel Xeon | Docker Compose    | [Deployment Guide](../deployment/docker_compose/README.md) |

## Validated Configurations

The following configurations have been validated for the language-detection microservice.

| **Deploy Method** | **Platform** |
| ----------------- | ------------ |
| Docker Compose    | Intel Xeon   |
