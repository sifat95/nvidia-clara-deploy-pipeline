# Clara Deploy Base Inference Application

## Overview

The NVIDIA Clara Train Transfer Learning Toolkit, TLT, for Medical Imaging provides
pre-trained models unique to medical imaging plus additional capabilities such as
integration with the AI-assisted Annotation SDK for speeding up annotation of medical images.
This allows the user to have access to AI-assisted labeling
[[Reference]](https://developer.nvidia.com/transfer-learning-toolkit).

In order to accelerate the deployment of TLT trained models using Clara Deploy SDK, this containerized
AI inference application was developed to be used as the base container which can be customized for
deploying specific TLT trained models. The customized container is then used as the AI inference operator in the Clara Deploy pipelines.

To customize the base container, configuration files used during model training and validation with
Clara Train TLT must be available. Also, the TLT trained model must have been exported using a format
compatible with TensorRT Inference Server.

This base inference application shares the same set of transform functions and the same scanning window
inference logic with Clara Train SDK V1.0, however, the output writer is Clara Deploy specific in order to support Clara Deploy pipeline results registration. If Clara Train SDK V2.0 has been used to train the model, the transform fucntions defined in the configuration file need to be translated to their counterpart in V1.0 before being used with this base inference application. 

This base application is included in the Clara Deploy SDK as a pre-built container. Steps on how to
create model specific application containers is provided in this section, and a sample application is also provided in the SDK.

### Version information

This base inference application is targeted to run in the following environment:
- Ubuntu 18.04
- Python 3.6
- NVIDIA TensorRT Inference Server Release 1.5.0, container version 19.08

## Inputs
This application, in the form of a Docker container, expects an input folder `/input` by default,
which can be mapped to the host volume when the Docker container is started. Expected in this
folder is a volume image file, of the format [NIfTI](https://nifti.nimh.nih.gov/) or [MetaImage](https://itk.org/Wiki/ITK/MetaIO/Documentation).
Further, it is expected that the volume image is constructed from a single series of a
DICOM study, typically the axial series with the data type of original primary.

## Outputs
This application saves the segmentation results to an output folder, `/output` by default,
which also can be mapped to a folder on the host volume. After the successful completion of the application, a segmentation volume image, of the format
[MetaImage](https://itk.org/Wiki/ITK/MetaIO/Documentation), is saved in the output folder.
The name of the output file is the same as that of the input file, due to certain limitations of the downstream operator in Clara Deploy SDK.

This container also publishes data for the Clara Deploy Render Server, in the `/publish` folder by default.
The original volume image, segmented volume image, along with config files for the Render Server are saved in this folder.

## AI Model
For testing, this base application uses a model trained using the NVIDIA Clara Train SDK V1.0 for liver tumor segmentation, namely `segmentation_liver_v1`. It is converted from TensorFlow Checkpoint model to `tensorflow_graphdef` using the Clara Train SDK V1.0 model export tool. The input tensor is of shape `96x96x96` with a single channel, and the output is of the same shape with 3 channels.

The key model attributes, e.g. the model name and network input dimensions, are saved in the `config_inference.json` file and consumed by this application at runtime.

### NVIDIA TensorRT Inference Server (TRTIS)
This application performs inference on the NVIDIA TensorRT Inference Server, Triton (formerly known as TRTIS), which
provides an inferencing solution optimized for NVIDIA GPUs.
The server provides an inference service via an HTTP or gRPC endpoint, allowing remote clients to request inferencing for any model being managed by the server. [Read more on Triton](https://docs.nvidia.com/deeplearning/sdk/triton-inference-server-guide/docs/).

## Directory Structure
The directories in the container are shown below.

```bash
/
├── sdk_dist/
└── app_base_inference
    ├── config
    │   ├── config_render.json
    │   ├── config_inference.json
    │   ├── __init__.py
    │   └── model_config.json
    ├── public
    │   └── docs
    │       └── README.md
    ├── writers
    │   ├── __init__.py
    |   ├── classification_result_writer.py
    │   ├── mhd_writer.py
    │   └── writer.py
    ├── app.py
    ├── Dockerfile
    ├── executor.py
    ├── logging_config.json
    ├── main.py
    ├── medical
    └── requirements.txt
/input
/output
/publish
/logs
```
The `sdk_dist` directory contains the Clara Train SDK V1.0 transforms library, whereas the `app_base_inference` contains the base application source code:
- The `config` directory contains model specific configuration files which need to be
replaced when building a customized container for a specific model.
  - The `config_inference.json` file contains the
configuration sections for pre and post transforms, model loader, referer, as well as writer.
  - The `model_config.json` contains the key attributes of the model.
  - The `config_render.json` contains configuration for the Clara Deploy Render Server.
- The `medical` directory contains compiled modules from Clara Train SDK.
- The `public` directory contains the documentation file(s).
- The `Writers` directory contains the specialized output writer required by Clara Deploy SDK, which saves the segmentation result to a volume image file as [MetaImage](https://itk.org/Wiki/ITK/MetaIO/Documentation).

If scanning window inferer is used, the model name must be correctly specified in the `inferer` attribute in the file `config_inference.json`, as shown in the following example

```javascript
    "inferer":
    {
        "name": "TRTISScanWindowInferer",
        "args": {
            "model_name": "segmentation_liver_v1",
            "ip": "localhost",
            "port": 8000,
            "protocol": "HTTP"
        }
    }
```

If the simple inferer is used, the model name and other attribute need to be specified in the file `model_config.json`.

```javascript
{
"model_name": "segmentation_liver_v1",
"input_node_names": ["cond/Merge"],
"output_node_names": ["NV_OUTPUT_LABEL"],
"data_format": "channels_first",
"output_channel_dict": {"NV_OUTPUT_LABEL": 3},
"network_input_size": [ 96, 96, 96]
}
```

## Executing Locally

To see the internals of the container, and to manually run the application, follow these steps. Please note, the TRTIS with the required model must be accessible from within this container, otherwise, failure will occur.

1. Start the container in interactive mode. See the next section on how to run the
container, but replace the `docker run` command with `docker run -it --entrypoint /bin/bash`
2. Once in the Docker terminal, ensure the current directory is `/`.
3. Type in command `python ./app_base_inference/main.py"`
4. Once finished, type `exit`.

## Executing in Docker

### Prerequisites
1. Ensure the Docker image of TRTIS has been imported into the local Docker repository by
 using the following command
 `docker images`
 and look for the image name of`TRTIS` and the correct tag for the release, e.g. `19.08-py3`.
2. Ensure that the model folder, including the config.pbtxt, is
present on the Clara Deploy host. Verify it by using the following steps:
   - Log on to the Clara Deploy host.
   - Check for the folder `liver_segmentation_v1` under the directory `/clara/common/models`.

### Step 1
Change to your working directory, e.g. `test`.

### Step 2
Create, if they do not exist, the following directories under your working directory:
  - `input` containing the input image file.
  - `output` for the segmentation output.
  - `publish` for publishing data for the Render Server.
  - `logs` for the log files.
  - `models` for models and copy over `liver_segmentation_v1` folder.

### Step 3
Note: If this base inference application container has already been pulled from NGC, please tag the container,
```bash
docker tag <pulled base container> app_base_inference:latest 
```
In your working directory, create a shell script, e.g. `run_base_docker.sh`, and copy the content below. Please comment or remove the command in the script that builds the container if it has been pulled from NGC. 
```bash
#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Clara Core would launch the container with the following environment variables internally,
# to provide runtime information.
export NVIDIA_CLARA_TRTISURI="localhost:8000"

APP_NAME="app_base_inference"
TRTIS_IMAGE="nvcr.io/nvidia/tensorrtserver:19.08-py3"

MODEL_NAME="segmentation_liver_v1"
NETWORK_NAME="container-demo"

# Install prerequisites.
# Note: Remove this command if the image has been pulled from NGC.
. envsetup.sh

# Create network
docker network create ${NETWORK_NAME}

# Run TRTIS(name: trtis), maping ./models/${MODEL_NAME} to /models/${MODEL_NAME}
# (localhost:8000 will be used)
nvidia-docker run --name trtis --network ${NETWORK_NAME} -d --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 \
    -v $(pwd)/models/${MODEL_NAME}:/models/${MODEL_NAME} ${TRTIS_IMAGE} \
    trtserver --model-store=/models

# Build Dockerfile.
# Note: Remove this command if the image has been pulled from NGC.
docker build -t ${APP_NAME} -f ${APP_NAME}/Dockerfile .

# Wait until TRTIS is ready
trtis_local_uri=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' trtis)
echo -n "Wait until TRTIS ${trtis_local_uri} is ready..."
while [ $(curl -s ${trtis_local_uri}:8000/api/status | grep -c SERVER_READY) -eq 0 ]; do
    sleep 1
    echo -n "."
done
echo "done"

export NVIDIA_CLARA_TRTISURI="${trtis_local_uri}:8000"

# Run ${APP_NAME} container.
# Like below, Clara Core would launch the app container with the following environment variables internally,
# to provide input/output path information.
# (They are subject to change. Do not use the environment variables directly in your application!)
docker run --name ${APP_NAME} --network ${NETWORK_NAME} -t --rm \
    -v $(pwd)/input:/input \
    -v $(pwd)/output:/output \
    -v $(pwd)/logs:/logs \
    -v $(pwd)/publish:/publish \
    -e NVIDIA_CLARA_TRTISURI \
    -e DEBUG_VSCODE \
    -e DEBUG_VSCODE_PORT \
    -e NVIDIA_CLARA_NOSYNCLOCK=TRUE \
    ${APP_NAME}

echo "${APP_NAME} has finished."

# Stop TRTIS container
echo "Stopping TRTIS"
docker stop trtis > /dev/null

# Remove network
docker network remove ${NETWORK_NAME} > /dev/null
```

### Step 4
Execute the created script, and wait for the application container to finish,
`./run_base_docker.sh`.

### Step 5
Check for the following output files:
- Original volume image (e.g. `image.mhd` and `image.raw`)
- Segmentation volume image (e.g. `image.out.mhd` and `image.out.raw`)
- Rendering configuration file (`config_render.json`)
- Metadata file describing the other files (`config.meta`)

## Creating Model Specific Application
This section describes how to use the base application container to build a model specific container
to deploy TLT trained models.

### Prerequisites
The user must first prepare data files using Clara Train TLT.
- With TLT Export tool, export the trained model to a platform compatible with TRTIS,
e.g. `tensorflow_graphdef`. The server side configuration file, config.pbtxt, must also be
generated. For details, please refer to
 [Triton](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/) and
 [Clara Train SDK](https://docs.nvidia.com/clara/index.html).
- A client side model configuration file, `model_config.json`, should also be prepared.
- The validation and inference pipeline configuration file must be available.
- A test data set of volume image, in [NIfTI](https://nifti.nimh.nih.gov/) or [MetaImage](https://itk.org/Wiki/ITK/MetaIO/Documentation) format, is available for testing the
container directly.
- A test data set of DICOM studies is available for testing Clara Deploy pipeline created
with the customized application as its inference operator.
### Steps
#### Step 1
Pull the base application container into the local Docker registry, if not already present.

#### Step 2
Create a Python project, e.g. `my_custom_app`, with the folder structure like below
```
my_custom_app
├── config
│   ├── config_inference.json
│   └── model_config.json
├── Dockerfile
└── public
    └── docs
        └── README.md
```
where the model_config.json contains the model attributes, and the config_inference.json can be
copied from the configuration used during training validation. This file will be modified in
the next step.
#### Step 3
Open the config_inference.json file. Keep the `pre_transforms` and
`post_transforms` sections as is, but you must change the `name` for `inferer` and
`model_loader` sections exactly as shown below. The `model_name` must be changed to the model used in the inference.
```javascript
    "inferer":
    {
        "name": "TRTISScanWindowInferer",
        "args": {
            "model_name": "segmentation_liver_v1",
            "ip": "localhost",
            "port": 8000,
            "protocol": "HTTP"
        }
    },

    "model_loader":
    {
        "name": "TRTISModelLoader",
        "args": {
            "model_spec_file_name": "{PBTXT_PATH}"
        }
    }
```
#### Step 4
Open the Dockerfile, and create and or update it with the content shown below. 
Note: Please update the actual `app_base_inference` container name and tag if they are different in your environment.

```bash
# Build upon the named base container; version tag can be used if known.
FROM app_base_inference:latest

# This is a well known folder in the base container. Please do not change it.
ENV BASE_NAME="app_base_inference"

# This is the name of the folder containing the config files; same as the app name.
ENV MY_APP_NAME="my_custom_app"

WORKDIR /app

# Copy configuration files to overwrite base defaults
COPY ./$MY_APP_NAME/config/* ./$BASE_NAME/config/
```
#### Step 5
Build the customized container with the following command, or run the command using shell script
```
APP_NAME="my_custom_app"
docker build -t ${APP_NAME} -f ${APP_NAME}/Dockerfile .
```
