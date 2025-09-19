# 🧾 Prompt Registry Microservice

The Prompt Registry microservice facilitates the storage and retrieval of users' preferred prompts by establishing a connection with the databases. This microservice is designed to seamlessly integrate with OPEA applications, enabling data persistence and efficient management of user's preferred prompts.

## 🛠️ Features

- **Store Prompt**: Save user's preferred prompt into database.
- **Retrieve Prompt**: Fetch prompt from database based on user, id or even a keyword search.
- **Delete Prompt**: Remove prompt from database.

## ⚙️ Deployment Options

To get detailed, step-by-step instructions on deploying the `prompt_registry` microservice, you should consult the deployment guide. This guide will walk you through all the necessary steps, from building the Docker images to configuring your environment and running the service.

| Platform | Deployment Method | Database | Link                                                             |
| -------- | ----------------- | -------- | ---------------------------------------------------------------- |
| CPU      | Docker            | ArangoDB | [Deployment Guide](./deployment/docker_compose/README_arango.md) |
| CPU      | Docker            | MongoDB  | [Deployment Guide](./deployment/docker_compose/README.md)        |
| CPU      | Docker            | Redis    | [Deployment Guide](./deployment/docker_compose/README_redis.md)  |
| CPU      | Docker Compose    | ArangoDB | [Deployment Guide](./deployment/docker_compose/README_arango.md) |
| CPU      | Docker Compose    | MongoDB  | [Deployment Guide](./deployment/docker_compose/README.md)        |
| CPU      | Docker Compose    | Redis    | [Deployment Guide](./deployment/docker_compose/README_redis.md)  |
