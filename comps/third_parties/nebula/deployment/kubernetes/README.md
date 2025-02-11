# Deploy NebulaGraph cluster (with Kubectl)

## 1. Install cert-manager

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.yaml
```

## 2. Install Nebula Operator

```bash
helm repo add nebula-operator https://vesoft-inc.github.io/nebula-operator/charts
helm repo update
helm install nebula-operator nebula-operator/nebula-operator --namespace=<namespace_name> --version=${chart_version}
```

## 3. Install and start NebulaGraph cluster

Choose between the Enterprise and Community configuration files base on your license. Edit the config with the proper storageClassName and run the command below:

```bash
kubectl create -f enterprise_edition.yaml
or
kubectl create -f community_edition.yaml
```

## 4. Connect to NebulaGraph databases

Once you've set up a NebulaGraph cluster using Nebula Operator on Kubernetes, you can connect to NebulaGraph databases both from inside the cluster and from external sources.

### 4.1. Connect to NebulaGraph databases from within a NebulaGraph cluster

Run the following command to check the IP of the Service:

```bash
$ kubectl get service -l app.kubernetes.io/cluster=<nebula>  #<nebula> is a variable value. Replace it with the desired name.
NAME                       TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)                                          AGE
nebula-graphd-svc          ClusterIP   10.98.213.34   <none>        9669/TCP,19669/TCP,19670/TCP                     23h
nebula-metad-headless      ClusterIP   None           <none>        9559/TCP,19559/TCP,19560/TCP                     23h
nebula-storaged-headless   ClusterIP   None           <none>        9779/TCP,19779/TCP,19780/TCP,9778/TCP
```

Run the following command to connect to the NebulaGraph database using the IP of the <cluster-name>-graphd-svc Service above:

```bash
kubectl run -ti --image vesoft/nebula-console --restart=Never -- <nebula_console_name> -addr <cluster_ip>  -port <service_port> -u <username> -p <password>
```

For example:

```bash
kubectl run -ti --image vesoft/nebula-console --restart=Never -- nebula-console -addr 10.98.213.34  -port 9669 -u root -p vesoft

```
A successful connection to the database is indicated if the following is returned:

If you don't see a command prompt, try pressing enter.

(root@nebula) [(none)]>

### 4.2. Connect to NebulaGraph databases from outside a NebulaGraph cluster

Refer to the [NebulaGraph Database Manual](https://docs.nebula-graph.io/3.1.3/nebula-operator/4.connect-to-nebula-graph-service/) for more information.
