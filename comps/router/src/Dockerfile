# comps/router/src/Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 6000

# TODO: This service relies on huggingface tokens, etc. – need to know best way to integrate.. via secrets or env etc

CMD ["python", "opea_router_microservice.py"]
