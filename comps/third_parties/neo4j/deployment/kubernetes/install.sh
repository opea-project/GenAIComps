helm repo add neo4j https://helm.neo4j.com/neo4j && helm repo update # call once
release_name=${1:-graph-neo}
version=${2:-5.23.0}
NS=ogpt
./uninstall.sh ${release_name}
helm install -n $NS --version $version ${release_name} neo4j/neo4j -f values.yaml

