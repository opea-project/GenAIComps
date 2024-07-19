# build predictionguard embeddings endpoint docker image

```
docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t opea/embedding-pg:latest -f comps/embeddings/predictionguard/docker/Dockerfile .
```

# launch predictionguard embedding endpoint docker container

```
docker run -d --name="embedding-pg-server" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p 6000:6000 --ipc=host -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY opea/embedding-pg:latest
```

# run client test

```
curl localhost:6000/v1/embeddings \
     -X POST \
     -d '{"text":"Hello, world!"}' \
     -H 'Content-Type: application/json'
```
