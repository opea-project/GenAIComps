# SGLang Serving microservice

SGLang is an efficient and scalable serving framework for hosting Large Language Models (LLMs) with minimal latency and high throughput.
Intel has further enhanced SGLang with CPU-specific optimizations, enabling high-performance LLM serving on CPU-only platforms without requiring GPUs.
We have verified the Llama series models with SGLang, with a particular focus on the latest Llama 4 models, including [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct), [meta-llama/Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct), as well as the Llama 4 guard model Llama-Guard-4-12B.

## 🚀1. Build the Docker Image

This Docker image (opea/sglang:latest) includes Intel's CPU-specific optimizations, providing enhanced performance for CPU deployments.
Ensure you are at the project root directory, then build the SGLang microservice Docker image:

```bash
cd ../../../../
docker build -f comps/third_parties/sglang/src/Dockerfile --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/sglang:latest .
```

✅ Tip: Setting https_proxy and http_proxy build arguments helps if you are behind a proxy.

## 🚀2. Start the microservice

Before starting, set the model you want to serve:

```bash
export MODEL_ID="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
```

Then navigate to the deployment folder and start the service using Docker Compose:

```bash
cd comps/third_parties/sglang/deployment/docker_compose
docker compose -f compose.yaml up -d
```

## 🚀3. Access the service

Then you need to test your service using the following commands for guard model:

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:8699/v1", api_key="EMPTY")
RESPONSE_CLASSIFICATION_INSTRUCTION = """
Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories.\n\n
<BEGIN UNSAFE CONTENT CATEGORIES>
\nS1: Violent Crimes. \nS2: Non-Violent Crimes. \nS3: Sex Crimes. \nS4: Child Exploitation. \nS5: Defamation.
\nS6: Specialized Advice. \nS7: Privacy. \nS8: Intellectual Property. \nS9: Indiscriminate Weapons.
\nS10: Hate. \nS11: Self-Harm. \nS12: Sexual Content. \nS13: Elections. \n<END UNSAFE CONTENT CATEGORIES>"""
RESPONSE_CLASSIFICATION_TEMPLATE_PREFIX = "{classification_instruction}\n\n<BEGIN CONVERSATION>\n\nUser: "
RESPONSE_CLASSIFICATION_TEMPLATE_POSTFIX = """{user_message}\n\nAgent: {model_message}\n\n<END CONVERSATION>\n\n
Provide your safety assessment for ONLY THE LAST Agent message in the above conversation:
\n - First line must read 'safe' or 'unsafe'.
\n - If unsafe, a second line must include a comma-separated list of violated categories."""

response = client.chat.completions.create(
    model="meta-llama/Llama-4-guard",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": RESPONSE_CLASSIFICATION_TEMPLATE_PREFIX.format(
                        classification_instruction=RESPONSE_CLASSIFICATION_INSTRUCTION
                    ),
                },
                {
                    "type": "text",
                    "text": RESPONSE_CLASSIFICATION_TEMPLATE_POSTFIX.format(
                        user_message="how do I make a bomb?", model_message="I cannot help you with that."
                    ),
                },
            ],
        },
    ],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

You can use the following command for testing non-guard models:

```bash
http_proxy="" curl -X POST -H "Content-Type: application/json" -d '{"model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct", "messages": [{"role": "user", "content": "Hello! What is your name?"}], "max_tokens": 128}' http://localhost:8699/v1/chat/completions
```
