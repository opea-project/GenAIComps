# Start ArangoDB Server

## 1. Download ArangoDB image

```bash
docker pull arangodb:latest
```

## 2. Configure the password

```bash
export ARANGO_ROOT_PASSWORD=${your_arango_password}
```

## 3. Run ArangoDB service

To launch ArangoDB locally, first ensure you have docker installed. Then, you can launch the database with the following docker command.

```bash
docker run -d --name arangodb -p 8529:8529 -e ARANGO_ROOT_PASSWORD=${ARANGO_ROOT_PASSWORD} arangodb/arangodb:latest --experimental-vector-index true
```
