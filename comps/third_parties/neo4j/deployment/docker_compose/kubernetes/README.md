# Deploy Neo4j on kubernetes cluster

- You should have Helm (version >= 3.15) installed. Refer to the [Helm Installation Guide](https://helm.sh/docs/intro/install/) for more information.
- For more deployment options, refer to [helm charts README](https://github.com/opea-project/GenAIInfra/tree/main/helm-charts#readme).

## Deploy on Xeon

```
helm repo add neo4j https://helm.neo4j.com/neo4j && helm repo update # call once
release_name=${1:-graph-neo}
version=${2:-5.23.0}
NS=ogpt
./uninstall.sh ${release_name}
helm install -n $NS --version $version ${release_name} neo4j/neo4j -f values.yaml
```
