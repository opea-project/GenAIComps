
docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume="/home/xianyang/xinyao/work/kgllm/OPEA/GenAIComps/comps/graphs/data":"/data" \
    --env='NEO4JLABS_PLUGINS=["apoc"]' \
    neo4j

docker run \
    --rm \
    --name="knowledge-graph-server" \
    -p 8060:8060 \
    --ipc=host \
    -v /home/xianyang/xinyao/work/kgllm/OPEA/GenAIComps/comps:/home/user/comps \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e NEO4J_ENDPOINT=$NEO4J_ENDPOINT \
    -e NEO4J_USERNAME=$NEO4J_USERNAME \
    -e NEO4J_PASSWORD=$NEO4J_PASSWORD \
    opea/knowledge_graphs:latest



curl http://localhost:8060/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'

curl http://localhost:8060/v1/graphs \
  -X POST \
  -d "{\"text\":\"MATCH (t:Task {status:'open'}) RETURN count(*)\",\"strtype\":\"cypher\"}" \
  -H 'Content-Type: application/json'
