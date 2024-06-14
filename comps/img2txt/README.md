# Image To Text Microservice (HPU)

This microservice use LLaVA as the base model. It basically accepts two inputs: a prompt and an image. It outputs the answer to the prompt about the image. It is widely used for Visual Question and Answering tasks.

# 🚀1. Start Microservice with Python (Option 1)

## 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

## 1.2 Start Image To Text Service/Test

* Xeon CPU

```bash
# Start LLaVA service
nohup python llava_server.py --device=cpu &
# Test
python check_llava_server.py
```

```bash
# Start the OPEA Microservice
python img2txt.py
# Test
python check_img2txt.py
```

* Gaudi2 HPU

```bash
# Start LLaVA service
nohup python llava_server.py &
# Test
python check_llava_server.py
```

```bash
# Start the OPEA Microservice
python img2txt.py
# Test
python check_img2txt.py
```