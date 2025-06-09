# Router Microservice

> Location: comps/router/src/README.md

A lightweight HTTP service that routes incoming text prompts to the most appropriate LLM back‑end (e.g. strong vs weak) and returns the target inference endpoint. It is built on the OPEA micro‑service SDK and can switch between two controller back‑ends:

- RouteLLM (matrix‑factorisation, dataset‑driven)
- Semantic‑Router (encoder‑based semantic similarity)

The router is stateless; it inspects the prompt, consults the configured controller, and replies with a single URL such as http://opea_router:8000/strong.

## Build

```bash
# From repo root 📂
# Build the container image directly
$ docker build -t opea/router:latest -f comps/router/src/Dockerfile .
```

Alternatively, the Docker Compose workflow below will build the image for you.

```bash
# Navigate to the compose bundle
$ cd comps/router/deployment/docker_compose

# Populate required secrets (or create a .env file)
$ export HF_TOKEN="<your hf token>"
$ export OPENAI_API_KEY="<your openai key>"

# Optional: point to custom inference endpoints / models
$ export WEAK_ENDPOINT=http://my‑llm‑gateway:8000/weak
$ export STRONG_ENDPOINT=http://my‑llm‑gateway:8000/strong
$ export CONTROLLER_TYPE=routellm        # or semantic_router

# Launch (using the helper script)
$ ./deploy_router.sh
```

_The service listens on http://localhost:6000 (host‑mapped from container port 6000). Logs stream to STDOUT; use Ctrl‑C to stop or docker compose down to clean up._

## RouteLLM compatibility patch

The upstream **RouteLLM** project is geared toward OpenAI embeddings and GPT-4–augmented
checkpoints.
We include a small patch – `hf_compatibility.patch` – that:

- adds a `hf_token` plumb-through,
- switches the Matrix-Factorisation router to Hugging Face sentence embeddings,
- removes hard-coded GPT-4 “golden-label” defaults.

**Container users:**
The Dockerfile applies the patch automatically during `docker build`, so you don’t have to do anything.

**Local development:**

```bash
# 1. Clone upstream RouteLLM
git clone https://github.com/lm-sys/RouteLLM.git
cd RouteLLM

# 2. Apply the patch shipped with this repo
patch -p1 < ../comps/router/src/hf_compatibility.patch

# 3. Install the patched library
pip install -e .
```

## API Usage

| Method | URL         | Body schema                   | Success response                               |
| ------ | ----------- | ----------------------------- | ---------------------------------------------- |
| `POST` | `/v1/route` | `{ "text": "<user prompt>" }` | `200 OK` → `{ "url": "<inference endpoint>" }` |

**Example**

```
curl -X POST http://localhost:6000/v1/route \
     -H "Content-Type: application/json" \
     -d '{"text": "Explain the Calvin cycle in photosynthesis."}'
```

Expected JSON _(assuming the strong model wins the routing decision)_:

```
{
  "url": "http://opea_router:8000/strong"
}
```

## Configuration Reference

| Variable / file                     | Purpose                                           | Default                                | Where set           |
| ----------------------------------- | ------------------------------------------------- | -------------------------------------- | ------------------- |
| `HF_TOKEN`                          | Hugging Face auth token for encoder models        | —                                      | `.env` / shell      |
| `OPENAI_API_KEY`                    | OpenAI key (only if `embedding_provider: openai`) | —                                      | `.env` / shell      |
| `CONTROLLER_TYPE`                   | `routellm` or `semantic_router`                   | `routellm`                             | env / `router.yaml` |
| `CONFIG_PATH`                       | Path to global router YAML                        | `/app/configs/router.yaml`             | Compose env         |
| `WEAK_ENDPOINT` / `STRONG_ENDPOINT` | Final inference URLs                              | container DNS                          | Compose env         |
| `WEAK_MODEL_ID` / `STRONG_MODEL_ID` | Model IDs forwarded to controllers                | `openai/gpt-3.5-turbo`, `openai/gpt-4` | Compose env         |

## Troubleshooting

`HF_TOKEN` is not set – export the token or place it in a .env file next to compose.yaml.

Unknown controller type – `CONTROLLER_TYPE` must be either routellm or semantic_router and a matching entry must exist in controller_config_paths.

Routed model `<name>` not in `model_map` – make sure model_map in router.yaml lists both strong and weak with the correct model_id values.

Use docker compose logs -f router_service for real‑time debugging.

## Testing

Includes an end-to-end script for the RouteLLM controller:

```bash
chmod +x tests/router/test_router_routellm.sh
export HF_TOKEN="<your HF token>"
export OPENAI_API_KEY="<your OpenAI key>"
tests/router/test_router_routellm.sh
```
