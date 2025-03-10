# Deploy OVMS on kubernetes cluster

- You should have Helm (version >= 3.15) installed. Refer to the [Helm Installation Guide](https://helm.sh/docs/intro/install/) for more information.
- For more deployment options, refer to [helm charts README](https://github.com/openvinotoolkit/operator/tree/main/helm-charts/ovms).

## Deploy on Xeon

```
git clone https://github.com/openvinotoolkit/operator
cd operator/tree/main/helm-charts/ovms
# set persistent volume claim with models repository created using [models export tools](https://github.com/openvinotoolkit/model_server/tree/main/demos/common/export_models)
export PVC=<enter persistent volume claim>
export CONFIG_PATH=config_all.json

cat > cpu-values.yaml << EOF
image_name: openvino/model_server:2025.0
deployment_parameters:
  replicas: 1
  openshift_service_mesh: false
service_parameters:
    grpc_port: 8080
    rest_port: 8081
    service_type: "ClusterIP"
models_settings:
  single_model_mode: false
  config_path=${CONFIG_PATH}
models_repository:
  models_host_path: ""
  models_volume_claim: ${PVC}
monitoring:
  metrics_enable: true
EOF
helm install ovms-app ovms -f cpu-values.yaml
```
