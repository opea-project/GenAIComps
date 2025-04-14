# Router Microservice Source

This folder contains the **Router microservice** code and integration logic for OPEA. The entrypoint is [`opea_router_microservice.py`](./opea_router_microservice.py).

---

## Code Structure

```
comps/router/src
├── README.md         # (this file)
├── opea_router_microservice.py
└── integrations/
    └── controllers/
        ├── base_controller.py
        ├── controller_factory.py
        ├── routellm_controller/
        │   └── routellm_controller.py
        └── semantic_router_controller/
            └── semantic_router_controller.py
```

### Key Files

- **`opea_router_microservice.py`**  
  - Registers the router microservice with `@register_microservice`.
  - Endpoint: `/v1/route` on port 6000.
  - Loads the config from `CONFIG_PATH` (defaults to `/app/config.yaml`).
  - Chooses a "controller" (RouteLLM or Semantic) based on that config.

- **`integrations/controllers/`**  
  - **`base_controller.py`**: Abstract base class.
  - **`controller_factory.py`**: Chooses which subclass to instantiate.
  - **`routellm_controller.py`**: Implements a learned gating approach (RouteLLM).
  - **`semantic_router_controller.py`**: Uses embedding-based routing with a `SemanticRouter`.

---

## How It Works

1. **Initialization**  
   - The microservice reads a global `config.yaml` (plus a specialized config like `routellm_config.yaml`) on startup.
   - The config references a `model_map` dict (e.g., "strong" → GPT-4, "weak" → GPT-3.5).

2. **Request Handling**  
   - When a user POSTs JSON to `/v1/route` with a `text` field, the service:
     1. Passes the user text to the chosen controller.
     2. The controller returns an `endpoint` from `model_map`.
     3. The microservice responds with `{ "url": "<endpoint>" }`.

3. **Runtime Environment**  
   - Typically run in Docker or Kubernetes, with environment variables for tokens.

---

## Example Microservice Call

If the service is running on `localhost:6000`:

```
curl -X POST http://localhost:6000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi, can you show me a geometric proof of the Pythagorean theorem?"}'
```

**Response** (example):
```
{
  "url": "http://some-inference-service:8000/strong"
}
```
indicating the "strong" model is more appropriate.

---

## Development Notes

- **Dependencies**: In production images, tokens (Hugging Face / OpenAI) must be set.
- **Logging**: Uses `CustomLogger` from `comps`.
- **Extensibility**: Add new controllers in `integrations/controllers/` and update `controller_factory.py`.

---

