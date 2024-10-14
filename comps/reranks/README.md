# ReRank Microservice

## Introduction

Provides functionality for reranking outputs.
It allows for fine-tuning the results based on specific criteria, improving the overall quality and relevance for the user.

## Key Features

- **Configurable Ranking Criteria**: Define custom ranking rules based on various factors like relevance, creativity, style, or user-defined metrics.
- **Multiple Ranking Models**: Integrate with different ranking models, enabling flexibility in selection based on the desired outcome and data type.
- **Real-time Reranking**: Rerank outputs in real-time, ensuring the best possible results are presented to the user with minimal latency.
- **Performance Monitoring**: Monitor the performance of the reranking process, including processing time and effectiveness of applied rules.
- **Scalability**: Designed to handle large-scale workloads and support increasing volumes of generated outputs.
- **Integration with Other Services**: Seamless integration with other microservices in the application for a smooth data flow and overall processing pipeline.

## Additional Notes

- This document provides a high-level overview of the Reranking Microservice.
- For detailed information on configuration, available ranking models, and API documentation, please refer to the dedicated service specifications.
- This microservice can be customized to specific application needs through configuration and integration with external tools.

## Users are able to configure and build reranking-related services according to their actual needs.

### Reranking Microservice with TEI

For details, please refer to [readme](tei/README.md).

### Reranking Microservice with Mosec

For details, please refer to this [readme](mosec/langchain/README.md).

### Reranking Microservice with FastRAG

For details, please refer to this [readme](fastrag/README.md).

### Reranking Microservice with VideoQnA

For details, please refer to this [readme](videoqna/README.md).
