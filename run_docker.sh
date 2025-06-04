#!/bin/bash

# build the docker image
docker build --debug -t usenix2025-forging .

# start and run the script to generate a lenet/mnist trace and to
# verify that trace
CONTAINER_NAME=usenix-forging-$(date +%Y%m%dT%H%M%S)

docker run \
    --name $CONTAINER_NAME \
    -it \
    --gpus all \
    usenix2025-forging \
    bash


# copy experiment results to local
docker cp $CONTAINER_NAME:/usenix2025/experiments/ ./

# remove the container
docker rm $CONTAINER_NAME
