# LVM Prediction Guard Microservice

Visual Question and Answering is one of the multimodal tasks empowered by LVMs (Large Visual Models). This microservice supports visual Q&A by using a LLaVA model available via the Prediction Guard API. It accepts two inputs: a prompt and an image. It outputs the answer to the prompt about the image.

# 🚀1. Start Microservice with Python

## 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

## 1.2 Start LVM Service

```bash
python lvm.py
```

# 🚀2. Start Microservice with Docker (Option 2)

## 2.1 Setup Environment Variables

Setup the following environment variables first

```bash
export PREDICTIONGUARD_API_KEY=${your_predictionguard_api_key}
```

## 2.1 Build Docker Images

```bash
cd ../..
docker build -t opea/lvm-pg:latest -f comps/lvms_pg/Dockerfile .
```

## 2.2 Start Service

```bash
docker run -d --name="lvm-pg" -p 8091:8091 -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY opea/lvm-pg:latest
```

# 🚀3. Consume LVM Service

```bash
curl -X POST http://localhost:8091/v1/lvm \
    -H 'Content-Type: application/json' \
    -d '{
      "image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC",
      "prompt": "What is this?",
      "max_new_tokens": 30
    }' 
```
