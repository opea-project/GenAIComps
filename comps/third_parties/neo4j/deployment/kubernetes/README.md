# Deploy Neo4j on kubernetes cluster

## Deploy on Xeon
Step 1: Edit values.yaml and replace the values of the following items with the actual configurations of your system:
- passwordFromSecret
- storageClassName

Step 2: Run the following command 
```
./install.sh ${release_name} $version $NS
```
