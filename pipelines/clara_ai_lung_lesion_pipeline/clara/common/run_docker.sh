#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Define the name of the app (aka operator), assumed the same as the project folder name
APP_NAME="lung_lesion_seg_app"

# Define the TenseorRT Inference Server Docker image, which will be used for testing
# Use either local repo or NVIDIA repo
TRTIS_IMAGE="nvcr.io/nvidia/tensorrtserver:20.12-py3"

# Launch the container with the following environment variables
# to provide runtime information.
export NVIDIA_CLARA_TRTISURI="localhost:8000"

# Define the model name for use when launching TRTIS with only the specific model
MODEL_NAME="segmentation_ct_lung_lesion_v1"

# Create a Docker network so that containers can communicate on this network
NETWORK_NAME="container-demo"

# Create network
sudo docker network create ${NETWORK_NAME}

# Run TRTIS(name: trtis), maping ./models/${MODEL_NAME} to /models/${MODEL_NAME}
# (localhost:8000 will be used)
sudo nvidia-docker run --name trtis --network ${NETWORK_NAME} -d --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 \
    -v $(pwd)/models/${MODEL_NAME}:/models/${MODEL_NAME} ${TRTIS_IMAGE} \
    trtserver --model-store=/models

# Wait until TRTIS is ready
trtis_local_uri=$(sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' trtis)
echo -n "Wait until TRTIS ${trtis_local_uri} is ready..."
while [ $(curl -s ${trtis_local_uri}:8000/api/status | grep -c SERVER_READY) -eq 0 ]; do
    sleep 1
    echo -n "."
done
echo "done"

export NVIDIA_CLARA_TRTISURI="${trtis_local_uri}:8000"

# Run ${APP_NAME} container.
# Launch the app container with the following environment variables internally,
# to provide input/output path information.
sudo docker run --name ${APP_NAME} --network ${NETWORK_NAME} -it --rm \
    -v $(pwd)/input:/input \
    -v $(pwd)/output:/output \
    -v $(pwd)/logs:/logs \
    -v $(pwd)/publish:/publish \
    -e NVIDIA_CLARA_TRTISURI \
    -e NVIDIA_CLARA_NOSYNCLOCK=TRUE \
    ${APP_NAME}

echo "${APP_NAME} is done."

# Stop TRTIS container
echo "Stopping TRTIS"
docker stop trtis > /dev/null

# Remove network
sudo docker network remove ${NETWORK_NAME} > /dev/null
