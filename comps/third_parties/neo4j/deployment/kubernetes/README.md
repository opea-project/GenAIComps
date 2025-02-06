# Deploy Neo4j on kubernetes cluster

- You should have Helm (version >= 3.15) installed. Refer to the [Helm Installation Guide](https://helm.sh/docs/intro/install/) for more information.
- For more deployment options, refer to [helm charts README](https://github.com/opea-project/GenAIInfra/tree/main/helm-charts#readme).

## Deploy on Xeon
Step 1: Edit values.yaml and replace the values of the following items with the actual configurations of your system:
- passwordFromSecret
- storageClassName

Step 2: Run the following command 
```
./install.sh ${release_name} $version $NS
```
