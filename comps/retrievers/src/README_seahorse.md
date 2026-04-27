# Retriever Microservice with Seahorse Cloud

## 0. Prerequisites

Before using this microservice, you need a Seahorse Cloud table with an API endpoint:

1. Sign up at [Seahorse Console](https://console.seahorse.dnotitia.ai)
2. Create a table — an API endpoint (`SEAHORSE_BASE_URL`) and API key (`SEAHORSE_API_KEY`) will be issued upon creation
3. Use the issued endpoint and key as environment variables below

## 1. 🚀 Start Microservice with Python (Option 1)

### 1.1 Install Requirements

```bash
cd ../../../
pip install -e .
cd comps/retrievers/src
pip install -r requirements-cpu.txt
```

### 1.2 Setup Environment Variables

```bash
export SEAHORSE_BASE_URL="https://<table-uuid>.api.seahorse.dnotitia.ai"
export SEAHORSE_API_KEY="sk_xxx"
export SEAHORSE_EMBEDDING_MODE="builtin"  # or "external"
export SEAHORSE_SEARCH_MODE="hybrid"      # "dense", "sparse", or "hybrid"
```

> **⚠️ Important**: `SEAHORSE_EMBEDDING_MODE` must be the same value for both Retriever and Dataprep services.
> If Dataprep used `builtin` mode to ingest documents, Retriever must also use `builtin`.
> If Dataprep used `external` mode to ingest documents, Retriever must also use `external`.

> **⚠️ SEAHORSE_SEARCH_MODE**: This must match the vector column configuration of your Seahorse table.
> For example, if your table only has a dense vector column, set `SEAHORSE_SEARCH_MODE="dense"`.
> Setting `hybrid` or `sparse` on a table without a sparse vector column will cause search failures.
> When `SEAHORSE_EMBEDDING_MODE="external"`, Retriever search is always dense-only.
> Set `SEAHORSE_SEARCH_MODE="dense"`; `hybrid` and `sparse` will be overridden to `dense`.

> Both `SEAHORSE_EMBEDDING_MODE` and `SEAHORSE_SEARCH_MODE` are normalized to lowercase and trimmed at startup, so values like `Builtin`, `HYBRID`, or `dense` are accepted. An unknown `SEAHORSE_SEARCH_MODE` raises `RuntimeError` at boot listing the supported values.

For `external` mode, you can also set:

```bash
export TEI_EMBEDDING_ENDPOINT="http://${your_ip}:6060"
export HF_TOKEN=${your_huggingface_token}
```

If `TEI_EMBEDDING_ENDPOINT` is not set, Retriever falls back to local HuggingFace embeddings to satisfy the SDK's external-mode initialization requirement.

### 1.3 Start Retriever Service

```bash
export RETRIEVER_COMPONENT_NAME="OPEA_RETRIEVER_SEAHORSE"
python opea_retrievers_microservice.py
```

## 2. 🚀 Start Microservice with Docker (Option 2)

### 2.1 Setup Environment Variables

```bash
export SEAHORSE_BASE_URL="https://<table-uuid>.api.seahorse.dnotitia.ai"
export SEAHORSE_API_KEY="sk_xxx"
export SEAHORSE_EMBEDDING_MODE="builtin"  # set "external" for dense-only vector search
export SEAHORSE_SEARCH_MODE="hybrid"      # set "dense" when SEAHORSE_EMBEDDING_MODE="external"
export TEI_EMBEDDING_ENDPOINT="http://${your_ip}:6060"
export HF_TOKEN=${your_huggingface_token}
export RETRIEVER_COMPONENT_NAME="OPEA_RETRIEVER_SEAHORSE"
```

### 2.2 Build Docker Image

```bash
cd ../../../
docker build -t opea/retriever:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/src/Dockerfile .
```

### 2.3 Run Docker with CLI

```bash
docker run -d --name="retriever-seahorse-server" -p 7000:7000 --ipc=host -e SEAHORSE_BASE_URL=$SEAHORSE_BASE_URL -e SEAHORSE_API_KEY=$SEAHORSE_API_KEY -e SEAHORSE_EMBEDDING_MODE=$SEAHORSE_EMBEDDING_MODE -e SEAHORSE_SEARCH_MODE=$SEAHORSE_SEARCH_MODE -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e HF_TOKEN=$HF_TOKEN -e RETRIEVER_COMPONENT_NAME=$RETRIEVER_COMPONENT_NAME opea/retriever:latest
```

## 🚀 3. Consume Retriever Service

### 3.1 Check Service Status

```bash
curl http://${your_ip}:7000/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
```

### 3.2 Search Documents

**builtin mode** (text query — server handles embedding):

```bash
curl http://${your_ip}:7000/v1/retrieval \
  -X POST \
  -d '{"text":"What is the revenue of Nike in 2023?","embedding":[0.0],"k":3}' \
  -H 'Content-Type: application/json'
```

**external mode** (pre-computed embedding vector, dense-only):

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://${your_ip}:7000/v1/retrieval \
  -X POST \
  -d "{\"text\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding},\"k\":3}" \
  -H 'Content-Type: application/json'
```

## 4. Embedding Modes

| Mode       | Env Var                            | Search Method                                                                                                                                                                   | TEI Required?                        |
| ---------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `builtin`  | `SEAHORSE_EMBEDDING_MODE=builtin`  | `similarity_search(query=text)` — server embeds query                                                                                                                           | No                                   |
| `external` | `SEAHORSE_EMBEDDING_MODE=external` | `similarity_search_by_vector(embedding=vec)` — dense-only query vector search. External embeddings apply to dense vectors only; sparse always uses the built-in embedding path. | No (falls back to local HuggingFace) |

## 5. Search Type Behavior

| `search_type`                   | Dense (`builtin` + `SEAHORSE_SEARCH_MODE=dense`, or `external`)                                                                        | Sparse / Hybrid (`builtin` only)                                                                                                            |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `similarity`                    | Top-`k` by cosine distance — no threshold applied                                                                                      | Top-`k` by similarity score — no threshold applied                                                                                          |
| `similarity_score_threshold`    | `score_threshold` is interpreted as a **distance upper bound** (lower = closer match). Returns docs with `distance ≤ score_threshold`. | `score_threshold` is interpreted as a **similarity lower bound** (higher = better match). Returns docs with `similarity ≥ score_threshold`. |
| `similarity_distance_threshold` | `distance_threshold` is the **distance upper bound**. Recommended for dense distance cutoffs.                                          | Not supported — returns HTTP 400.                                                                                                           |
| `mmr`                           | **Not supported.** Silently falls back to `similarity` (a warning is logged). For diversity-aware retrieval, use a different backend.  | Same as dense.                                                                                                                              |

> **Important:** `score_threshold` semantics differ between dense and sparse/hybrid because the underlying SDK returns distances for dense vector search and similarities for sparse/hybrid. Apply the threshold accordingly when comparing results across modes. When in doubt for dense searches, prefer `similarity_distance_threshold` so the intent is unambiguous.
