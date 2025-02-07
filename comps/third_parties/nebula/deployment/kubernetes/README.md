# Deploy NebulaGraph cluster (with Kubectl)

## 1. Install cert-manager

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.yaml
```

## 2. Install Nebula Operator

```bash
<<<<<<< HEAD
helm repo add nebula-operator https://vesoft-inc.github.io/nebula-operator/charts
helm repo update
=======
helm repo add nebula-operator https://vesoft-inc.github.io/nebula-operator/charts
helm repo update
>>>>>>> origin/nebula
helm install nebula-operator nebula-operator/nebula-operator --namespace=<namespace_name> --version=${chart_version}
```

## 3. Install and start NebulaGraph clusyter

Choose the between Enterprise and Community configuration file base on your license. Edit the config with the proper storageClassName and run the command below:
<<<<<<< HEAD
=======

> > > > > > > origin/nebula

```bash
kubectl create -f enterprise_edition.yaml
or
kubectl create -f community_edition.yaml
```

<<<<<<< HEAD

=======

> > > > > > > origin/nebula
