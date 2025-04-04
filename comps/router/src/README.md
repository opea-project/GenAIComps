# Router Microservice (src)

This directory contains the core implementation of the **LLM Router Microservice** adapted for OPEA's GenAIComps microservice architecture. It dynamically routes user queries to optimal inference endpoints based on configurable logic.

## Key Files & Structure

```
src/
├── integrations/
│   └── controllers/
│       ├── routellm/
│       │   ├── config.yaml
│       │   └── routellm_controller.py
│       ├── semantic_router/
│       │   ├── config.yaml
│       │   └── semantic_router_controller.py
│       ├── base_controller.py
│       └── controller_factory.py
├── opea_router_microservice.py       # Main OPEA-registered microservices
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## How `opea_router_microservice.py` Works

Defines and registers three OPEA-compliant microservice endpoints:

- `/v1/route`: Determines the best inference endpoint for a given query.
- `/v1/route-forward`: Routes the query, forwards it, and returns the response.
- `/v1/route/reload-config`: Dynamically reloads configuration without restart.

## Controllers

- **Semantic Router** (`semantic_router_controller.py`): Routes based on semantic similarity to preset queries.
- **RouteLLM** (`routellm_controller.py`): Uses ML methods (e.g., Matrix Factorization) to choose optimal models dynamically.

Controllers and their configs are bundled within the Docker image. The active controller is selected via the global `config.yaml`.

## Configuration

Set environment variable `CONFIG_PATH` to point to your main `config.yaml`, defining:

```yaml
model_map:
  weak:
    endpoint: "http://service:8000/weak"
    model_id: "MODEL_ID_2"
  strong:
    endpoint: "http://service:8000/strong"
    model_id: "MODEL_ID_1"

controller_config_path: "integrations/controllers/routellm/config.yaml"
```

Environment variables needed:
- `HF_TOKEN`: For Hugging Face embeddings

## Running & Testing

- Use Docker Compose or Kubernetes deployments defined in the `deployment` directory.
- Endpoint testing example:

```bash
curl -X POST http://localhost:6000/v1/route \
     -H "Content-Type: application/json" \
     -d '{"text": "Your query here"}'
```

This migration retains routing flexibility while aligning with OPEA’s modular, microservice-focused architecture.

