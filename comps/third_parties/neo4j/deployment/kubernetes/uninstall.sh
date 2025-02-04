release_name=${1:-graph-neo}
NS=ogpt

tsleep=0
helm uninstall -n $NS ${release_name} && tsleep=5
neo_pvc=( $(kubectl get pvc -l helm.neo4j.com/instance=${release_name} --no-headers -o custom-columns=CONTAINER:.metadata.name))
for pvc in "${neo_pvc[@]}"; do
    kubectl delete pvc $pvc
done
sleep $tsleep
