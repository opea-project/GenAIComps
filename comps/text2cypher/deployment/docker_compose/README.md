# Deploy text2cypher microservice using docker-compose

## Deploy on Intel Gaudi

```
unset http_proxy
service_name="neo4j-apoc text2cypher-gaudi"
export ip_address=$(hostname -I | awk '{print $1}')
export host_ip=${ip_address}
export TAG="comps"
export NEO4J_AUTH="neo4j/neo4jtest"
export NEO4J_URL="bolt://${ip_address}:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="neo4jtest"
export NEO4J_apoc_export_file_enabled=true
export NEO4J_apoc_import_file_use__neo4j__config=true
export NEO4J_PLUGINS=\[\"apoc\"\]

cd $WORKPATH/comps/text2cypher/deployment/docker_compose/
docker compose up ${service_name} -d

```
