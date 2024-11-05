# # Copyright (C) 2024 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os
from typing import Any, Dict, List

import yaml
from exporter import convert_args_to_command, replace_env_vars
from kubernetes import client

log_level = os.getenv("LOGLEVEL", "INFO")
logging.basicConfig(level=log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")


def load_service_info(file_path=None):
    if file_path:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return data

    return None


def create_k8s_resources(
    name,
    image,
    container_ports,
    node_selector={"node-type": "opea"},
    container_name=None,
    namespace="default",
    replicas=1,
    app_label=None,
    topology_spread_constraints=None,
    args=None,
    env=None,
    env_from=[client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name="qna-config"))],
    resources=None,
    volumes=None,
    volume_mounts=None,
    annotations={"sidecar.istio.io/rewriteAppHTTPProbers": "true"},
    security_context=None,
    host_ipc=True,
    image_pull_policy="IfNotPresent",
):

    if app_label is None and container_name is None:
        app_label = name
        container_name = name

    topology_spread_constraints = [
        client.V1TopologySpreadConstraint(
            max_skew=1,
            topology_key="kubernetes.io/hostname",
            when_unsatisfiable="ScheduleAnyway",
            label_selector=client.V1LabelSelector(match_labels={"app": app_label}),
        )
    ]

    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        spec=client.V1DeploymentSpec(
            replicas=replicas,
            selector=client.V1LabelSelector(match_labels={"app": app_label}),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(annotations=annotations, labels={"app": app_label}),
                spec=client.V1PodSpec(
                    node_selector=node_selector,
                    topology_spread_constraints=topology_spread_constraints,
                    host_ipc=host_ipc,
                    service_account_name="default",
                    containers=[
                        client.V1Container(
                            name=container_name,
                            image=image,
                            image_pull_policy=image_pull_policy,
                            args=args,
                            ports=[client.V1ContainerPort(container_port=p) for p in container_ports],
                            env_from=env_from if env_from is not None else None,
                            env=env if env is not None else None,
                            resources=resources,
                            volume_mounts=volume_mounts,
                            security_context=security_context,
                        )
                    ],
                    volumes=volumes,
                ),
            ),
        ),
    )

    return deployment


def create_resource_requirements(limits=None, requests=None):
    """Create a V1ResourceRequirements object with optional limits and requests.

    :param limits: A dictionary of resource limits, e.g., {"cpu": "4", "memory": "8Gi"}
    :param requests: A dictionary of resource requests, e.g., {"cpu": "2", "memory": "4Gi"}
    :return: A V1ResourceRequirements object
    """
    return client.V1ResourceRequirements(
        limits=limits if limits is not None else None, requests=requests if requests is not None else None
    )


def create_configmap_object(config_dict=None, config_name="qna-config"):

    if config_dict is None:
        config_map = {
            "EMBEDDING_MODEL_ID": "BAAI/bge-base-en-v1.5",
            "RERANK_MODEL_ID": "BAAI/bge-reranker-base",
            "LLM_MODEL_ID": "Intel/neural-chat-7b-v3-3",
            "TEI_EMBEDDING_ENDPOINT": "http://embedding-dependency-svc.default.svc.cluster.local:6006",
            # For dataprep only
            "TEI_ENDPOINT": "http://embedding-dependency-svc.default.svc.cluster.local:6006",
            # For dataprep & retrieval & vector_db
            "INDEX_NAME": "rag-redis",
            "REDIS_URL": "redis://vector-db.default.svc.cluster.local:6379",
            "TEI_RERANKING_ENDPOINT": "http://reranking-dependency-svc.default.svc.cluster.local:8808",
            "TGI_LLM_ENDPOINT": "http://llm-dependency-svc.default.svc.cluster.local:9009",
            "HUGGINGFACEHUB_API_TOKEN": "${HF_TOKEN}",
            "EMBEDDING_SERVICE_HOST_IP": "embedding-svc",
            "RETRIEVER_SERVICE_HOST_IP": "retriever-svc",
            "RERANK_SERVICE_HOST_IP": "reranking-svc",
            "NODE_SELECTOR": "chatqna-opea",
            "LLM_SERVICE_HOST_IP": "llm-svc",
        }
    else:
        config_map = config_dict

    configmap = client.V1ConfigMap(
        api_version="v1",
        kind="ConfigMap",
        metadata=client.V1ObjectMeta(name=config_name, namespace="default"),
        data=config_map,
    )
    return configmap


def create_no_wrapper_configmap_object(service_info=None):

    if service_info is None:
        config_map = {
            "EMBEDDING_MODEL_ID": "BAAI/bge-base-en-v1.5",
            "RERANK_MODEL_ID": "BAAI/bge-reranker-base",
            "LLM_MODEL_ID": "Intel/neural-chat-7b-v3-3",
            "TEI_EMBEDDING_ENDPOINT": "http://embedding-dependency-svc.default.svc.cluster.local:6006",
            # For dataprep only
            "TEI_ENDPOINT": "http://embedding-dependency-svc.default.svc.cluster.local:6006",
            # For dataprep & retrieval & vector_db
            "INDEX_NAME": "rag-redis",
            "REDIS_URL": "redis://vector-db.default.svc.cluster.local:6379",
            "TEI_RERANKING_ENDPOINT": "http://reranking-dependency-svc.default.svc.cluster.local:8808",
            "TGI_LLM_ENDPOINT": "http://llm-dependency-svc.default.svc.cluster.local:9009",
            "HUGGINGFACEHUB_API_TOKEN": "${HF_TOKEN}",
            "EMBEDDING_SERVER_HOST_IP": "embedding-dependency-svc",
            "RETRIEVER_SERVICE_HOST_IP": "retriever-svc",
            "RERANK_SERVER_HOST_IP": "reranking-dependency-svc",
            "NODE_SELECTOR": "chatqna-opea",
            "LLM_SERVER_HOST_IP": "llm-dependency-svc",
        }
    else:
        configmap = service_info

    configmap = client.V1ConfigMap(
        api_version="v1",
        kind="ConfigMap",
        metadata=client.V1ObjectMeta(name="qna-config", namespace="default"),
        data=config_map,
    )
    return configmap


def create_service(name, app_label, service_ports, namespace="default", service_type="ClusterIP"):
    ports = []
    for port in service_ports:
        # Create a dictionary mapping to handle different field names
        port_dict = {
            "name": port.get("name"),
            "port": port.get("port"),
            "target_port": port.get("target_port"),
            "node_port": port.get("nodePort"),  # Map 'nodePort' to 'node_port'
        }

        # Remove keys with None values
        port_dict = {k: v for k, v in port_dict.items() if v is not None}

        ports.append(client.V1ServicePort(**port_dict))

    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        spec=client.V1ServiceSpec(type=service_type, selector={"app": app_label}, ports=ports),
    )
    return service


def create_embedding_deployment_and_service(resource_requirements=None, replicas=1, image_name=None, args=None):

    args = ["--model-id", "$(EMBEDDING_MODEL_ID)", "--auto-truncate"]
    volume_mounts = [
        client.V1VolumeMount(name="model-volume", mount_path="/data"),
        client.V1VolumeMount(name="shm", mount_path="/dev/shm"),
    ]

    volumes = [
        client.V1Volume(
            name="model-volume",
            host_path=client.V1HostPathVolumeSource(path="/mnt/models", type="Directory"),
        ),
        client.V1Volume(name="shm", empty_dir=client.V1EmptyDirVolumeSource(medium="Memory", size_limit="1Gi")),
    ]

    deployment = create_k8s_resources(
        name="embedding-dependency-resources",
        replicas=replicas,
        app_label="embedding-dependency-resources",
        image="ghcr.io/huggingface/text-embeddings-inference:cpu-1.5",
        container_name="embedding-dependency-resources",
        container_ports=[80],
        node_selector={"node-type": "chatqna-opea"},
        resources=resource_requirements,
        env_from=[client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name="qna-config"))],
        args=args,
        volume_mounts=volume_mounts,
        volumes=volumes,
    )

    embedding_dependency_ports = [
        {
            "name": "service",
            "port": 6006,
            "target_port": 80,
        },
    ]
    service = create_service(
        name="embedding-dependency-svc",
        app_label="embedding-dependency-resources",
        service_ports=embedding_dependency_ports,
    )

    return deployment, service


def create_embedding_svc_deployment_and_service(resource_requirements=None, replicas=1, image_name=None, args=None):

    deployment = create_k8s_resources(
        name="embedding-resources",
        replicas=replicas,
        image="opea/embedding-tei:latest",
        container_ports=[6000],
        resources=resource_requirements,
        env_from=[client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name="qna-config"))],
    )

    ports = [
        {
            "name": "service",
            "port": 6000,
            "target_port": 6000,
        },
    ]
    service = create_service(name="embedding-svc", app_label="embedding-resources", service_ports=ports)

    return deployment, service


def create_llm_dependency_deployment_and_service(resource_requirements=None, replicas=1, image_name=None, args=None):

    if args is None:
        args = [
            "--model-id",
            "$(LLM_MODEL_ID)",
            "--max-input-length",
            "2048",
            "--max-total-tokens",
            "4096",
            "--max-batch-total-tokens",
            "65536",
            "--max-batch-prefill-tokens",
            "4096",
        ]

    volume_mounts = [
        client.V1VolumeMount(mount_path="/data", name="model-volume"),
        client.V1VolumeMount(mount_path="/dev/shm", name="shm"),
    ]

    env = [
        client.V1EnvVar(name="OMPI_MCA_btl_vader_single_copy_mechanism", value="none"),
        client.V1EnvVar(name="PT_HPU_ENABLE_LAZY_COLLECTIVES", value="true"),
        client.V1EnvVar(name="runtime", value="habana"),
        client.V1EnvVar(name="HABANA_VISIBLE_DEVICES", value="all"),
        client.V1EnvVar(name="HF_TOKEN", value="${HF_TOKEN}"),
    ]

    volumes = [
        client.V1Volume(
            name="model-volume",
            host_path=client.V1HostPathVolumeSource(path="/mnt/models", type="Directory"),
        ),
        client.V1Volume(name="shm", empty_dir=client.V1EmptyDirVolumeSource(medium="Memory", size_limit="1Gi")),
    ]

    security_context = client.V1SecurityContext(capabilities=client.V1Capabilities(add=["SYS_NICE"]))
    deployment = create_k8s_resources(
        name="llm-dependency-resources",
        replicas=replicas,
        image="ghcr.io/huggingface/tgi-gaudi:2.0.4",
        container_ports=[80],
        node_selector={"node-type": "chatqna-opea"},
        resources=resource_requirements,
        env_from=[client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name="qna-config"))],
        env=env,
        args=args,
        volume_mounts=volume_mounts,
        volumes=volumes,
        security_context=security_context,
    )

    ports = [
        {
            "name": "service",
            "port": 9009,
            "target_port": 80,
        },
    ]
    service = create_service(name="llm-dependency-svc", app_label="llm-dependency-resources", service_ports=ports)

    return deployment, service


def create_reranking_dependency_deployment_and_service(
    resource_requirements=None, replicas=1, image_name=None, args=None
):

    args = ["--model-id", "$(RERANK_MODEL_ID)", "--auto-truncate"]

    volume_mounts = [
        client.V1VolumeMount(mount_path="/data", name="model-volume"),
        client.V1VolumeMount(mount_path="/dev/shm", name="shm"),
    ]

    env = [
        client.V1EnvVar(name="OMPI_MCA_btl_vader_single_copy_mechanism", value="none"),
        client.V1EnvVar(name="PT_HPU_ENABLE_LAZY_COLLECTIVES", value="true"),
        client.V1EnvVar(name="runtime", value="habana"),
        client.V1EnvVar(name="HABANA_VISIBLE_DEVICES", value="all"),
        client.V1EnvVar(name="HF_TOKEN", value="${HF_TOKEN}"),
        client.V1EnvVar(name="MAX_WARMUP_SEQUENCE_LENGTH", value="512"),
    ]

    volumes = [
        client.V1Volume(
            name="model-volume",
            host_path=client.V1HostPathVolumeSource(path="/mnt/models", type="Directory"),
        ),
        client.V1Volume(name="shm", empty_dir=client.V1EmptyDirVolumeSource(medium="Memory", size_limit="1Gi")),
    ]

    volume_mounts = [
        client.V1VolumeMount(mount_path="/data", name="model-volume"),
        client.V1VolumeMount(mount_path="/dev/shm", name="shm"),
    ]

    deployment = create_k8s_resources(
        name="reranking-dependency-resources",
        replicas=replicas,
        image="opea/tei-gaudi:latest",
        container_ports=[80],
        node_selector={"node-type": "chatqna-opea"},
        resources=resource_requirements,
        env_from=[client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name="qna-config"))],
        env=env,
        args=args,
        volume_mounts=volume_mounts,
        volumes=volumes,
    )

    ports = [
        {
            "name": "service",
            "port": 8808,
            "target_port": 80,
        },
    ]
    service = create_service(
        name="reranking-dependency-svc", app_label="reranking-dependency-resources", service_ports=ports
    )

    return deployment, service


def create_llm_deployment_and_service(resource_requirements=None, replicas=1, image_name=None, args=None):

    deployment = create_k8s_resources(
        name="llm-resources",
        replicas=replicas,
        image="opea/llm-tgi:latest",
        container_ports=[9000],
        resources=resource_requirements,
    )

    ports = [
        {
            "name": "service",
            "port": 9000,
            "target_port": 9000,
        },
    ]
    service = create_service(name="llm-svc", app_label="llm-resources", service_ports=ports)

    return deployment, service


def create_dataprep_deployment_and_service(resource_requirements=None, replicas=1, image_name=None, args=None):
    deployment = create_k8s_resources(
        name="dataprep-resources",
        namespace="default",
        replicas=replicas,
        app_label="dataprep-resources",
        image="opea/dataprep-redis:latest",
        container_name="dataprep-resources",
        container_ports=[6007],
        node_selector={"node-type": "chatqna-opea"},
        resources=resource_requirements,
    )

    ports = [{"name": "port1", "port": 6007, "target_port": 6007}]
    service = create_service(name="dataprep-svc", app_label="dataprep-resources", service_ports=ports)

    return deployment, service


def create_chatqna_mega_deployment(resource_requirements=None, replicas=1, image_name=None, args=None):

    deployment = create_k8s_resources(
        name="chatqna-backend-server-resources",
        replicas=replicas,
        app_label="chatqna-backend-server-resources",
        image=image_name,
        container_name="chatqna-backend-server-resources",
        container_ports=[8888],
        node_selector={"node-type": "chatqna-opea"},
        resources=resource_requirements,
        env_from=[client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name="qna-config"))],
    )

    ports = [
        {"name": "service", "port": 8888, "target_port": 8888, "nodePort": 30888},
    ]
    service = create_service(
        name="chatqna-backend-server-svc",
        app_label="chatqna-backend-server-resources",
        service_type="NodePort",
        service_ports=ports,
    )

    return deployment, service


def create_reranking_deployment_and_service(resource_requirements=None, replicas=1, image_name=None, args=None):
    deployment = create_k8s_resources(
        name="reranking-resources",
        replicas=replicas,
        image="opea/reranking-tei:latest",
        container_ports=[8000],
        resources=resource_requirements,
    )

    ports = [
        {
            "name": "service",
            "port": 8000,
            "target_port": 8000,
        },
    ]
    service = create_service(name="reranking-svc", app_label="reranking-resources", service_ports=ports)

    return deployment, service


def create_retriever_deployment_and_service(resource_requirements=None, replicas=1, image_name=None, args=None):

    deployment = create_k8s_resources(
        name="retriever-resources",
        replicas=replicas,
        image="opea/retriever-redis:latest",
        container_ports=[7000],
        resources=resource_requirements,
    )

    ports = [
        {
            "name": "service",
            "port": 7000,
            "target_port": 7000,
        },
    ]
    service = create_service(name="retriever-svc", app_label="retriever-resources", service_ports=ports)

    return deployment, service


def create_vector_db_deployment_and_service(resource_requirements=None, replicas=1, image_name=None, args=None):

    deployment = create_k8s_resources(
        name="vector-db",
        replicas=replicas,
        image="redis/redis-stack:7.2.0-v9",
        container_ports=[6379, 8001],
        resources=resource_requirements,
    )

    ports = [
        {"name": "vector-db-service", "port": 6379, "target_port": 6379},
        {"name": "vector-db-insight", "port": 8001, "target_port": 8001},
    ]
    service = create_service(name="vector-db", app_label="vector-db", service_ports=ports)

    return deployment, service


def kubernetes_obj_to_dict(k8s_obj):
    return client.ApiClient().sanitize_for_serialization(k8s_obj)


def save_to_yaml(manifests_list, file_name):
    with open(file_name, "a") as f:
        for manifests in manifests_list:
            yaml.dump(kubernetes_obj_to_dict(manifests), f, default_flow_style=False)
            f.write("---\n")


def build_chatqna_manifests(service_info=None, output_filename=None):
    configmap = create_configmap_object(service_info)

    guaranteed_resource = create_resource_requirements(
        limits={"cpu": 8, "memory": "8000Mi"}, requests={"cpu": 8, "memory": "8000Mi"}
    )

    burstable_resource = create_resource_requirements(requests={"cpu": 4, "memory": "4000Mi"})

    # Microservice
    chatqna_deploy, chatqna_svc = create_chatqna_mega_deployment(guaranteed_resource, image_name="opea/chatqna:latest")
    embedding_deploy, embedding_deploy_svc = create_embedding_svc_deployment_and_service(burstable_resource)
    reranking_svc, reranking_svc_svc = create_reranking_deployment_and_service(burstable_resource)
    lm_deploy, lm_deploy_svc = create_llm_deployment_and_service(burstable_resource)

    # Embedding, Reranking and LLM
    embedding_dependency_resource = create_resource_requirements(
        limits={"cpu": 80, "memory": "20000Mi"}, requests={"cpu": 80, "memory": "20000Mi"}
    )
    embedding_dependency, embedding_dependency_svc = create_embedding_deployment_and_service(
        embedding_dependency_resource
    )

    llm_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    llm_dependency, llm_dependency_svc = create_llm_dependency_deployment_and_service(llm_hpu_cards)

    reranking_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    reranking_dependency, reranking_dependency_svc = create_reranking_dependency_deployment_and_service(
        reranking_hpu_cards
    )

    retrieval_deployment, retrieval_svc = create_retriever_deployment_and_service(burstable_resource)
    vector_db_deploy, vector_db_svc = create_vector_db_deployment_and_service()
    dataprep_deploy, dataprep_svc = create_dataprep_deployment_and_service()

    manifests = [
        configmap,
        chatqna_deploy,
        chatqna_svc,
        dataprep_deploy,
        dataprep_svc,
        embedding_dependency,
        embedding_dependency_svc,
        embedding_deploy,
        embedding_deploy_svc,
        llm_dependency,
        llm_dependency_svc,
        lm_deploy,
        lm_deploy_svc,
        reranking_dependency,
        reranking_dependency_svc,
        reranking_svc,
        reranking_svc_svc,
        retrieval_deployment,
        retrieval_svc,
        vector_db_deploy,
        vector_db_svc,
    ]

    save_to_yaml(manifests, output_filename)


def build_tuned_chatqna_manifests_with_rerank(
    service_info=None, output_filename=None, tgi_replicas=1, embedding_replicas=1, service_replicas=1
):
    configmap = create_configmap_object(service_info)

    guaranteed_resource = create_resource_requirements(
        limits={"cpu": 8, "memory": "8000Mi"}, requests={"cpu": 8, "memory": "8000Mi"}
    )

    burstable_resource = create_resource_requirements(requests={"cpu": 4, "memory": "4000Mi"})

    tgi_args = [
        "--model-id",
        "$(LLM_MODEL_ID)",
        "--max-input-length",
        "1024",
        "--max-total-tokens",
        "2048",
        "--max-batch-total-tokens",
        "65536",
        "--max-batch-prefill-tokens",
        "4096",
    ]

    chatqna_deploy, chatqna_svc = create_chatqna_mega_deployment(
        guaranteed_resource, image_name="opea/chatqna:latest", replicas=service_replicas
    )
    embedding_deploy, embedding_deploy_svc = create_embedding_svc_deployment_and_service(
        burstable_resource, replicas=service_replicas
    )
    reranking_svc, reranking_svc_svc = create_reranking_deployment_and_service(
        burstable_resource, replicas=service_replicas
    )
    lm_deploy, lm_deploy_svc = create_llm_deployment_and_service(burstable_resource, replicas=service_replicas)
    retrieval_deployment, retrieval_svc = create_retriever_deployment_and_service(
        burstable_resource, replicas=service_replicas
    )

    # Embedding, Reranking and LLM
    embedding_dependency_resource = create_resource_requirements(
        limits={"cpu": 80, "memory": "20000Mi"}, requests={"cpu": 80, "memory": "20000Mi"}
    )
    embedding_dependency, embedding_dependency_svc = create_embedding_deployment_and_service(
        embedding_dependency_resource, replicas=embedding_replicas
    )

    llm_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    llm_dependency, llm_dependency_svc = create_llm_dependency_deployment_and_service(
        llm_hpu_cards, replicas=tgi_replicas, args=tgi_args
    )

    reranking_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    reranking_dependency, reranking_dependency_svc = create_reranking_dependency_deployment_and_service(
        reranking_hpu_cards
    )

    # Others
    vector_db_deploy, vector_db_svc = create_vector_db_deployment_and_service()
    dataprep_deploy, dataprep_svc = create_dataprep_deployment_and_service()

    manifests = [
        configmap,
        chatqna_deploy,
        chatqna_svc,
        dataprep_deploy,
        dataprep_svc,
        embedding_dependency,
        embedding_dependency_svc,
        embedding_deploy,
        embedding_deploy_svc,
        llm_dependency,
        llm_dependency_svc,
        lm_deploy,
        lm_deploy_svc,
        reranking_dependency,
        reranking_dependency_svc,
        reranking_svc,
        reranking_svc_svc,
        retrieval_deployment,
        retrieval_svc,
        vector_db_deploy,
        vector_db_svc,
    ]

    save_to_yaml(manifests, output_filename)


def build_tuned_chatqna_manifests_without_rerank(
    service_info=None, output_filename=None, tgi_replicas=1, embedding_replicas=1, service_replicas=1
):
    configmap = create_configmap_object(service_info)

    guaranteed_resource = create_resource_requirements(
        limits={"cpu": 8, "memory": "8000Mi"}, requests={"cpu": 8, "memory": "8000Mi"}
    )

    burstable_resource = create_resource_requirements(requests={"cpu": 4, "memory": "4000Mi"})

    tgi_args = [
        "--model-id",
        "$(LLM_MODEL_ID)",
        "--max-input-length",
        "1024",
        "--max-total-tokens",
        "2048",
        "--max-batch-total-tokens",
        "65536",
        "--max-batch-prefill-tokens",
        "4096",
    ]

    chatqna_deploy, chatqna_svc = create_chatqna_mega_deployment(
        guaranteed_resource, image_name="opea/chatqna-without-rerank:latest", replicas=service_replicas
    )
    embedding_deploy, embedding_deploy_svc = create_embedding_svc_deployment_and_service(
        burstable_resource, replicas=service_replicas
    )
    lm_deploy, lm_deploy_svc = create_llm_deployment_and_service(burstable_resource, replicas=service_replicas)
    retrieval_deployment, retrieval_svc = create_retriever_deployment_and_service(
        burstable_resource, replicas=service_replicas
    )

    # Embedding, Reranking and LLM
    embedding_dependency_resource = create_resource_requirements(
        limits={"cpu": 80, "memory": "20000Mi"}, requests={"cpu": 80, "memory": "20000Mi"}
    )
    embedding_dependency, embedding_dependency_svc = create_embedding_deployment_and_service(
        embedding_dependency_resource, replicas=embedding_replicas
    )

    llm_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    llm_dependency, llm_dependency_svc = create_llm_dependency_deployment_and_service(
        llm_hpu_cards, replicas=tgi_replicas, args=tgi_args
    )

    # Others
    vector_db_deploy, vector_db_svc = create_vector_db_deployment_and_service()
    dataprep_deploy, dataprep_svc = create_dataprep_deployment_and_service()

    manifests = [
        configmap,
        chatqna_deploy,
        chatqna_svc,
        dataprep_deploy,
        dataprep_svc,
        embedding_dependency,
        embedding_dependency_svc,
        embedding_deploy,
        embedding_deploy_svc,
        llm_dependency,
        llm_dependency_svc,
        lm_deploy,
        lm_deploy_svc,
        retrieval_deployment,
        retrieval_svc,
        vector_db_deploy,
        vector_db_svc,
    ]

    save_to_yaml(manifests, output_filename)


def build_oob_chatqna_manifests_without_rerank(
    service_info=None, output_filename=None, tgi_replicas=1, embedding_replicas=1, service_replicas=1
):
    configmap = create_configmap_object(service_info)

    tgi_args = [
        "--model-id",
        "$(LLM_MODEL_ID)",
        "--max-input-length",
        "2048",
        "--max-total-tokens",
        "4096",
        "--max-batch-total-tokens",
        "65536",
        "--max-batch-prefill-tokens",
        "4096",
    ]

    chatqna_deploy, chatqna_svc = create_chatqna_mega_deployment(
        image_name="opea/chatqna-without-rerank:latest", replicas=service_replicas
    )
    embedding_deploy, embedding_deploy_svc = create_embedding_svc_deployment_and_service(replicas=service_replicas)
    lm_deploy, lm_deploy_svc = create_llm_deployment_and_service(replicas=service_replicas)
    retrieval_deployment, retrieval_svc = create_retriever_deployment_and_service(replicas=service_replicas)

    # Embedding, Reranking and LLM
    embedding_dependency, embedding_dependency_svc = create_embedding_deployment_and_service(
        replicas=embedding_replicas
    )

    llm_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    llm_dependency, llm_dependency_svc = create_llm_dependency_deployment_and_service(
        llm_hpu_cards, replicas=tgi_replicas, args=tgi_args
    )

    # Others
    vector_db_deploy, vector_db_svc = create_vector_db_deployment_and_service()
    dataprep_deploy, dataprep_svc = create_dataprep_deployment_and_service()

    manifests = [
        configmap,
        chatqna_deploy,
        chatqna_svc,
        dataprep_deploy,
        dataprep_svc,
        embedding_dependency,
        embedding_dependency_svc,
        embedding_deploy,
        embedding_deploy_svc,
        llm_dependency,
        llm_dependency_svc,
        lm_deploy,
        lm_deploy_svc,
        retrieval_deployment,
        retrieval_svc,
        vector_db_deploy,
        vector_db_svc,
    ]

    save_to_yaml(manifests, output_filename)


def build_oob_chatqna_manifests_with_rerank(
    service_info=None, output_filename=None, tgi_replicas=1, embedding_replicas=1
):
    configmap = create_configmap_object(service_info)

    # Microservice
    chatqna_deploy, chatqna_svc = create_chatqna_mega_deployment(image_name="opea/chatqna:latest")
    embedding_deploy, embedding_deploy_svc = create_embedding_svc_deployment_and_service()
    reranking_svc, reranking_svc_svc = create_reranking_deployment_and_service()
    lm_deploy, lm_deploy_svc = create_llm_deployment_and_service()

    embedding_dependency, embedding_dependency_svc = create_embedding_deployment_and_service()

    llm_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    llm_dependency, llm_dependency_svc = create_llm_dependency_deployment_and_service(
        llm_hpu_cards, replicas=tgi_replicas
    )

    reranking_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    reranking_dependency, reranking_dependency_svc = create_reranking_dependency_deployment_and_service(
        reranking_hpu_cards
    )

    retrieval_deployment, retrieval_svc = create_retriever_deployment_and_service()
    vector_db_deploy, vector_db_svc = create_vector_db_deployment_and_service()
    dataprep_deploy, dataprep_svc = create_dataprep_deployment_and_service()

    manifests = [
        configmap,
        chatqna_deploy,
        chatqna_svc,
        dataprep_deploy,
        dataprep_svc,
        embedding_dependency,
        embedding_dependency_svc,
        embedding_deploy,
        embedding_deploy_svc,
        llm_dependency,
        llm_dependency_svc,
        lm_deploy,
        lm_deploy_svc,
        reranking_dependency,
        reranking_dependency_svc,
        reranking_svc,
        reranking_svc_svc,
        retrieval_deployment,
        retrieval_svc,
        vector_db_deploy,
        vector_db_svc,
    ]

    save_to_yaml(manifests, output_filename)


def build_no_wrapper_tuned_chatqna_manifests_with_rerank(
    service_info=None, output_filename=None, tgi_replicas=1, embedding_replicas=1, service_replicas=1
):
    configmap = create_no_wrapper_configmap_object(service_info)

    guaranteed_resource = create_resource_requirements(
        limits={"cpu": 8, "memory": "8000Mi"}, requests={"cpu": 8, "memory": "8000Mi"}
    )

    burstable_resource = create_resource_requirements(requests={"cpu": 4, "memory": "4000Mi"})

    embedding_dependency_resource = create_resource_requirements(
        limits={"cpu": 80, "memory": "20000Mi"}, requests={"cpu": 80, "memory": "20000Mi"}
    )

    tgi_args = [
        "--model-id",
        "$(LLM_MODEL_ID)",
        "--max-input-length",
        "1280",
        "--max-total-tokens",
        "2048",
        "--max-batch-total-tokens",
        "65536",
        "--max-batch-prefill-tokens",
        "4096",
    ]

    # Microservice
    chatqna_deploy, chatqna_svc = create_chatqna_mega_deployment(
        guaranteed_resource, replicas=service_replicas, image_name="opea/chatqna-no-wrapper:latest"
    )

    embedding_dependency, embedding_dependency_svc = create_embedding_deployment_and_service(
        embedding_dependency_resource, replicas=embedding_replicas
    )

    llm_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    llm_dependency, llm_dependency_svc = create_llm_dependency_deployment_and_service(
        llm_hpu_cards, replicas=tgi_replicas, args=tgi_args
    )

    reranking_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    reranking_dependency, reranking_dependency_svc = create_reranking_dependency_deployment_and_service(
        reranking_hpu_cards
    )

    retrieval_deployment, retrieval_svc = create_retriever_deployment_and_service(
        burstable_resource, replicas=service_replicas
    )
    vector_db_deploy, vector_db_svc = create_vector_db_deployment_and_service()
    dataprep_deploy, dataprep_svc = create_dataprep_deployment_and_service()

    manifests = [
        configmap,
        chatqna_deploy,
        chatqna_svc,
        dataprep_deploy,
        dataprep_svc,
        embedding_dependency,
        embedding_dependency_svc,
        llm_dependency,
        llm_dependency_svc,
        reranking_dependency,
        reranking_dependency_svc,
        retrieval_deployment,
        retrieval_svc,
        vector_db_deploy,
        vector_db_svc,
    ]

    save_to_yaml(manifests, output_filename)


def build_no_wrapper_tuned_chatqna_manifests_without_rerank(
    service_info=None, output_filename=None, tgi_replicas=1, embedding_replicas=1, service_replicas=1
):
    configmap = create_no_wrapper_configmap_object(service_info)

    guaranteed_resource = create_resource_requirements(
        limits={"cpu": 8, "memory": "8000Mi"}, requests={"cpu": 8, "memory": "8000Mi"}
    )

    burstable_resource = create_resource_requirements(requests={"cpu": 4, "memory": "4000Mi"})

    embedding_dependency_resource = create_resource_requirements(
        limits={"cpu": 80, "memory": "20000Mi"}, requests={"cpu": 80, "memory": "20000Mi"}
    )

    tgi_args = [
        "--model-id",
        "$(LLM_MODEL_ID)",
        "--max-input-length",
        "1280",
        "--max-total-tokens",
        "2048",
        "--max-batch-total-tokens",
        "65536",
        "--max-batch-prefill-tokens",
        "4096",
    ]

    # Microservice
    chatqna_deploy, chatqna_svc = create_chatqna_mega_deployment(
        guaranteed_resource, replicas=service_replicas, image_name="opea/chatqna-no-wrapper-without-rerank:latest"
    )

    embedding_dependency, embedding_dependency_svc = create_embedding_deployment_and_service(
        embedding_dependency_resource, replicas=embedding_replicas
    )

    llm_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    llm_dependency, llm_dependency_svc = create_llm_dependency_deployment_and_service(
        llm_hpu_cards, replicas=tgi_replicas, args=tgi_args
    )

    retrieval_deployment, retrieval_svc = create_retriever_deployment_and_service(
        burstable_resource, replicas=service_replicas
    )
    vector_db_deploy, vector_db_svc = create_vector_db_deployment_and_service()
    dataprep_deploy, dataprep_svc = create_dataprep_deployment_and_service()

    manifests = [
        configmap,
        chatqna_deploy,
        chatqna_svc,
        dataprep_deploy,
        dataprep_svc,
        embedding_dependency,
        embedding_dependency_svc,
        llm_dependency,
        llm_dependency_svc,
        retrieval_deployment,
        retrieval_svc,
        vector_db_deploy,
        vector_db_svc,
    ]

    save_to_yaml(manifests, output_filename)


def build_no_wrapper_oob_chatqna_manifests_with_rerank(
    service_info=None, output_filename=None, tgi_replicas=1, embedding_replicas=1, service_replicas=1
):
    configmap = create_no_wrapper_configmap_object(service_info)

    # Microservice
    chatqna_deploy, chatqna_svc = create_chatqna_mega_deployment(image_name="opea/chatqna-no-wrapper:latest")

    embedding_dependency, embedding_dependency_svc = create_embedding_deployment_and_service()

    llm_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    llm_dependency, llm_dependency_svc = create_llm_dependency_deployment_and_service(
        llm_hpu_cards, replicas=tgi_replicas
    )

    reranking_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    reranking_dependency, reranking_dependency_svc = create_reranking_dependency_deployment_and_service(
        reranking_hpu_cards
    )

    retrieval_deployment, retrieval_svc = create_retriever_deployment_and_service()
    vector_db_deploy, vector_db_svc = create_vector_db_deployment_and_service()
    dataprep_deploy, dataprep_svc = create_dataprep_deployment_and_service()

    manifests = [
        configmap,
        chatqna_deploy,
        chatqna_svc,
        dataprep_deploy,
        dataprep_svc,
        embedding_dependency,
        embedding_dependency_svc,
        llm_dependency,
        llm_dependency_svc,
        reranking_dependency,
        reranking_dependency_svc,
        retrieval_deployment,
        retrieval_svc,
        vector_db_deploy,
        vector_db_svc,
    ]

    save_to_yaml(manifests, output_filename)


def build_no_wrapper_oob_chatqna_manifests_without_rerank(
    service_info=None, output_filename=None, tgi_replicas=1, embedding_replicas=1, service_replicas=1
):
    configmap = create_no_wrapper_configmap_object(service_info)

    # Microservice
    chatqna_deploy, chatqna_svc = create_chatqna_mega_deployment(
        image_name="opea/chatqna-no-wrapper-without-rerank:latest"
    )

    embedding_dependency, embedding_dependency_svc = create_embedding_deployment_and_service()

    llm_hpu_cards = create_resource_requirements(limits={"habana.ai/gaudi": 1})
    llm_dependency, llm_dependency_svc = create_llm_dependency_deployment_and_service(
        llm_hpu_cards, replicas=tgi_replicas
    )

    retrieval_deployment, retrieval_svc = create_retriever_deployment_and_service()
    vector_db_deploy, vector_db_svc = create_vector_db_deployment_and_service()
    dataprep_deploy, dataprep_svc = create_dataprep_deployment_and_service()

    manifests = [
        configmap,
        chatqna_deploy,
        chatqna_svc,
        dataprep_deploy,
        dataprep_svc,
        embedding_dependency,
        embedding_dependency_svc,
        llm_dependency,
        llm_dependency_svc,
        retrieval_deployment,
        retrieval_svc,
        vector_db_deploy,
        vector_db_svc,
    ]

    save_to_yaml(manifests, output_filename)


def extract_service_configs(input_data: Dict) -> Dict:

    all_configs = {}

    global_envs = input_data.get("global_envs", {})
    all_configs["config_map"] = global_envs

    all_services = [{**s, "type": "mega_service"} for s in input_data.get("mega_service", [])] + [
        {**s, "type": "micro_service"} for s in input_data.get("micro_services", [])
    ]

    for service in all_services:
        service_name = service["service_name"]
        service_config = {
            "image": service.get("image", None),
            "ports": service.get("ports", None),
            "volumes": service.get("volumes", None),
            "node_ports": service.get("node_ports", None),
            # "envs": service.get("envs", None),
            "type": service.get("type", None),
        }

        # for env in service.get("envs", []):
        #     if isinstance(env, list) and len(env) == 2:
        #         import pdb;pdb.set_trace()
        #         service_config["envs"][env[0]] = env[1]
        #     elif isinstance(env, dict):
        #         service_config["envs"].update(env)

        # print("envs = ", service_config.get("envs", None))
        # if "node_ports" in service:
        #     all_configs['node_ports'] = service["node_ports"]

        if "envs" in service:
            result_dict = {k: str(v) for d in service["envs"] for k, v in d.items()}
            service_config["envs"] = result_dict
            all_configs["config_map"].update(result_dict)

        service_config["replicas"] = service.get("replcias", 1)
        if "resources" in service:
            print("resources", service["resources"])

            resources = service.get("resources", {})
            requests = {}

            if "hpu" in service["resources"]:
                service["limits"] = {"habana.ai/gaudi": 1}

            if resources.get("cpu"):
                requests["cpu"] = resources["cpu"]
                service_config["resources"] = {"requests": requests}
            if resources.get("memory"):
                requests["memory"] = resources["memory"]
                service_config["resources"] = {"requests": requests}
            if resources.get("hpu"):
                requests["habana.ai/gaudi"] = resources["hpu"]
                service_config["resources"] = {"limits": requests}
            print("service_configresources", service["resources"])

        if "options" in service:
            for option in service["options"]:
                for key, value in option.items():
                    if key == "cap_add":
                        service_config[key] = [value] if isinstance(value, str) else value
                    else:
                        service_config[key] = value

        if "args" in service:
            service_args_list = []
            for item in service["args"]:
                if isinstance(item, dict):
                    for key, value in item.items():
                        service_args_list.extend([key, str(value)])
                else:
                    service_args_list.append(item)
            service_config["args"] = service_args_list

        print(f"{service_name} = {service_config} \n\n\n")
        all_configs[service_name] = service_config

    return all_configs


def create_deployment_and_service(
    service_name,
    ports,
    replicas=1,
    volume_mounts=None,
    volumes=None,
    args_list=None,
    resource_requirements=None,
    image_name=None,
    service_type="ClusterIP",
    args=None,
):

    microservice_deloyment_name = service_name + "-deploy"

    target_ports = list(set([port["target_port"] for port in ports]))

    deployment = create_k8s_resources(
        name=microservice_deloyment_name,
        image=image_name,
        replicas=replicas,
        app_label=microservice_deloyment_name,
        container_name=microservice_deloyment_name,
        container_ports=target_ports,
        node_selector={"node-type": "opea"},
        args=args_list,
        volume_mounts=volume_mounts,
        volumes=volumes,
        resources=resource_requirements,
        env_from=[client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name="qna-config"))],
    )

    service = create_service(
        name=service_name,
        app_label=microservice_deloyment_name,
        service_type=service_type,
        service_ports=ports,
    )

    return deployment, service


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and parse JSON/YAML files and output JSON file")

    args = parser.parse_args()

    with open("/home/zhenzhong/repo/opea/forked/yaoqing/GenAIComps/tests/cores/mega/mega.yaml", "r") as file:
        input_data = yaml.safe_load(file)

    input_data = replace_env_vars(input_data)
    all_configs = extract_service_configs(input_data)

    def build_deployment_and_service(all_configs, output_file="e2e_manifest.yaml"):
        config_dict = all_configs.get("config_map", None)

        config_map = create_configmap_object(config_dict=config_dict)
        save_to_yaml([config_map], output_file)

        all_manifests = []
        for service_name, service_config in all_configs.items():
            if service_name == "config_map":
                continue

            image = service_config.get("image", None)
            ports = service_config.get("ports", None)
            node_ports = service_config.get("node_ports", None)
            volumes_path = service_config.get("volumes", None)
            replicas = service_config.get("replicas", None)
            resources = service_config.get("resources", None)
            envs = service_config.get("envs", None)
            options = service_config.get("options", None)
            service_args = service_config.get("args", None)
            type = service_config.get("type", None)

            if type == "micro_service":
                service_type = "ClusterIP"
            elif type == "mega_service":
                service_type = "NodePort"

            # if ports is not None:
            #     formatted_ports = [{
            #         "name": f"port{i+1}",
            #         "port": int(p.split(":")[0]),
            #         "target_port": int(p.split(":")[1]),
            #     } for i, p in enumerate(ports)]

            if ports is not None:
                formatted_ports = [
                    {
                        "name": f"port{i+1}",
                        "port": int(p.split(":")[0]),
                        "target_port": int(p.split(":")[1]),
                        **({"nodePort": int(node_ports[i])} if node_ports and i < len(node_ports) else {}),
                    }
                    for i, p in enumerate(ports)
                ]

                # if node_ports is not None:
                #     import pdb;pdb.set_trace()
                # formatted_ports[i]["nodePort"]=int(node_ports[0])
                logging.debug(f"{formatted_ports}")

            logging.debug(f"{service_args}")
            logging.debug(f"{volumes_path}")
            logging.debug(f"envs = {envs}")

            allocated_resources = None
            if resources is not None:
                if resources.get("limits", None):
                    allocated_resources = create_resource_requirements(limits=resources["limits"])
                else:
                    allocated_resources = create_resource_requirements(requests=resources["requests"])

            volumes = None
            volume_mounts = None
            if volumes_path:
                volumes = []
                volume_mounts = []

                # Process each path in the input list
                for i, item in enumerate(volumes_path):
                    src, dest = item.split(":")

                    # Create volume for the source path
                    volumes.append(
                        client.V1Volume(
                            name=f"volume{i+1}",
                            host_path=client.V1HostPathVolumeSource(path=src, type="Directory"),
                        )
                    )

                    # Create volume mount for the destination path
                    volume_mounts.append(client.V1VolumeMount(name=f"volume{i+1}", mount_path=dest))

                volumes.append(
                    client.V1Volume(
                        name="shm", empty_dir=client.V1EmptyDirVolumeSource(medium="Memory", size_limit="1Gi")
                    )
                )
                volume_mounts.append(client.V1VolumeMount(name="shm", mount_path="/dev/shm"))

            env = [
                client.V1EnvVar(name="OMPI_MCA_btl_vader_single_copy_mechanism", value="none"),
                client.V1EnvVar(name="PT_HPU_ENABLE_LAZY_COLLECTIVES", value="true"),
                client.V1EnvVar(name="runtime", value="habana"),
                client.V1EnvVar(name="HABANA_VISIBLE_DEVICES", value="all"),
                client.V1EnvVar(name="HF_TOKEN", value="${HF_TOKEN}"),
            ]

            deployment, service = create_deployment_and_service(
                service_name=service_name,
                image_name=image,
                args_list=service_args,
                replicas=replicas,
                resource_requirements=allocated_resources,
                volumes=volumes,
                volume_mounts=volume_mounts,
                ports=formatted_ports,
                service_type=service_type,
            )
            all_manifests = [deployment, service]
            save_to_yaml(all_manifests, output_file)

        # save_to_yaml(all_manifests, 'all_test.yaml')

    build_deployment_and_service(all_configs, output_file="all_test.yaml")

    # build_no_wrapper_oob_chatqna_manifests_with_rerank(service_info,
    #                                                    tgi_replicas=7,
    #                                                    output_filename="no_wrapper_single_gaudi_with_rerank.yaml")

    # build_no_wrapper_tuned_chatqna_manifests_with_rerank(
    #     service_info,
    #     tgi_replicas=7,
    #     embedding_replicas=1,
    #     service_replicas=2,
    #     output_filename="no_wrapper_tuned_single_gaudi_with_rerank.yaml",
    # )

    # build_no_wrapper_oob_chatqna_manifests_without_rerank(
    #     service_info, tgi_replicas=7, output_filename="no_wrapper_single_gaudi_without_rerank.yaml"
    # )

    # build_no_wrapper_tuned_chatqna_manifests_without_rerank(
    #     service_info,
    #     tgi_replicas=8,
    #     embedding_replicas=1,
    #     service_replicas=2,
    #     output_filename="no_wrapper_tuned_single_gaudi_without_rerank.yaml",
    # )
