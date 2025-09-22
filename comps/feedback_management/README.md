# 🗨 Feedback Management Microservice

The Feedback Management microservice facilitates the storage and retrieval of users'feedback data by establishing a connection with the databases. This microservice is designed to seamlessly integrate with OPEA applications, enabling data persistence and efficient management of feedback data.

## 🛠️ Features

- **Store Feedback**: Save feedback data from user into database.
- **Retrieve Feedback**: Fetch feedback data from database based on user or id.
- **Update Feedback**: Update feedback data info in the database based on id.
- **Delete Feedback**: Remove feedback record from database.

## ⚙️ Deployment Options

To get detailed, step-by-step instructions on deploying the `feedback_management` microservice, you should consult the deployment guide. This guide will walk you through all the necessary steps, from building the Docker images to configuring your environment and running the service.

| Platform | Deployment Method | Database | Link                                                             |
| -------- | ----------------- | -------- | ---------------------------------------------------------------- |
| CPU      | Docker            | ArangoDB | [Deployment Guide](./deployment/docker_compose/README_arango.md) |
| CPU      | Docker            | MongoDB  | [Deployment Guide](./deployment/docker_compose/README.md)        |
| CPU      | Docker            | Redis    | [Deployment Guide](./deployment/docker_compose/README_redis.md)  |
| CPU      | Docker Compose    | MongoDB  | [Deployment Guide](./deployment/docker_compose/README.md)        |
| CPU      | Docker Compose    | ArangoDB | [Deployment Guide](./deployment/docker_compose/README_arango.md) |
| CPU      | Docker Compose    | Redis    | [Deployment Guide](./deployment/docker_compose/README_redis.md)  |
