# OPEA Router Component

This **Router** component provides logic to decide which inference endpoint or LLM best suits a given user request. It can run in different modes, such as:
- **RouteLLM** mode using a learned gating approach (e.g., "strong" vs "weak" model).
- **Semantic** mode using embeddings to route queries by semantic similarity.

---

## Getting Started

1. **Deployment**: See [deployment/README.md](./deployment/README.md) for Docker Compose or Kubernetes instructions.  
2. **Tests**: See [tests/README.md](./tests/README.md) for how to run local integration tests.  
3. **Microservice Code**: [src/README.md](./src/README.md) details the code structure and endpoints.

