# Start Neo4J Server

## 1. Download Neo4J image

```bash
docker pull neo4j:latest
```

## 2. Configure the username, password and dbname

```bash
export NEO4J_AUTH=neo4j/password
export NEO4J_PLUGINS=\[\"apoc\"\]
```

## 3. Run Neo4J service

To launch Neo4j locally, first ensure you have docker installed. Then, you can launch the database with the following docker command.

```bash
docker run \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/data -v $PWD/plugins:/plugins \
    --name neo4j-apoc \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS=\[\"apoc\"\]  \
    neo4j:latest
```
