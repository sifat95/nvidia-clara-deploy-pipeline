# Build upon the named base container; version tag can be used if known.
FROM nvcr.io/nvidia/clara/app_base_inference_v2:0.7.3-2011.5

# This is a well known folder in the base container. Please do not change it.
ENV BASE_NAME="app_base_inference_v2"

# This is the name of the folder containing the config files; same as the app name.
ENV MY_APP_NAME="lung_lesion_seg_app"

# RUN pip install -r requirements.txt
WORKDIR ./

COPY . ./$BASE_NAME/

RUN pip install -r ./$BASE_NAME/requirements.txt

