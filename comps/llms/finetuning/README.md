# Start Funetuning Service Locally

## Start Ray

```bash
$ ./start-ray-for-funetuning.sh
```

## Start Finetuning Service

```bash
$ ./run.sh
```

## Browse FastAPI Web UI for Experiments

http://localhost:8000/docs

### Sample Request for Creating Finetuning Job

```json
{
  "training_file": "file-vGxE9KywnSUkEL6dv9qZxKAF.jsonl",
  "model": "meta-llama/Llama-2-7b-chat-hf"
}
```

# Dev Notes

### Test if Ray cluster is working

```bash
$ python -c "import ray; ray.init(); print(ray.cluster_resources())"
```
