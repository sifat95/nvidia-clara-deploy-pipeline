# nvidia-clara-deploy-pipeline

# lung_leson_seg_app

It is to build the containerized infernece app based on [this](https://ngc.nvidia.com/catalog/containers/nvidia:clara:app_base_inference_v2)

Build the container by

```bash
$ cd lung_lesion_seg_app
$ docker build -t lung_lesion_seg_app .
```

# pipelines

This folder contains the pipeline configuration files.

Keep the model configuration in the 'clara/common/models' in the .pbtxt format and model in folder named 1.