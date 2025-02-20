# Deploy OVMS on kubernetes cluster

- You should have Helm (version >= 3.15) installed. Refer to the [Helm Installation Guide](https://helm.sh/docs/intro/install/) for more information.
- For more deployment options, refer to [helm charts README](https://github.com/openvinotoolkit/operator/tree/main/helm-charts/ovms).


## Deploy on Xeon

```
git clone https://github.com/openvinotoolkit/operator
cd operator/tree/main/helm-charts/ovms
# set persistent volume claim with models repository created using [models export tools](https://github.com/openvinotoolkit/model_server/tree/main/demos/common/export_models)
export PVC=
export CONFIG_PATH=
helm install ovms-app ovms --set global.models_settings.config_path=${CONFIG_PATH} --set global.models_repository.models_volume_claim=${PVC} -f cpu-values.yaml
```

