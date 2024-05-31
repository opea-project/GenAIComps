#!/bin/bash


cd docker

docker build \
    -f Dockerfile ../../ \
    -t ray_serve:habana \
    --network=host \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    --build-arg no_proxy=${no_proxy}
