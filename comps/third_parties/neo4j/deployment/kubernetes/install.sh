# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

helm repo add neo4j https://helm.neo4j.com/neo4j && helm repo update # call once
release_name=${1:-graph-neo}
version=${2:-5.23.0}
NS=${3:-ogpt}
./uninstall.sh ${release_name} $NS
helm install -n $NS --version $version ${release_name} neo4j/neo4j -f values.yaml
