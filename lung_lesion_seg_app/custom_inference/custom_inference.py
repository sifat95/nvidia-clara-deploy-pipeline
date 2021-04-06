import logging

import torch
import numpy as np
from skimage import util
#from .exp_configs import exp_configs

from .src import models
from torchvision import transforms
from haven import haven_utils as hu

from ai4med.common.medical_image import MedicalImage
from ai4med.common.transform_ctx import TransformContext
#from nvmidl.apps.aas.configs.modelconfig import ModelConfig
#from nvmidl.apps.aas.inference.inference import Inference
#from nvmidl.apps.aas.inference.inference_utils import InferenceUtils

class CustomInference():

    def inference(self, data):
        logger = logging.getLogger(__name__)
        logger.info('Run Custom Inference for: {}'.format(name))

        # Better to use Clara Transformers (extended based on the same to take max benefits of MedicalImage)
        assert isinstance(data, TransformContext)

        transform_ctx: TransformContext = data
        img: MedicalImage = transform_ctx.get_image('image')

        shape_fmt = img.get_shape_format()
        logger.info('Shape Format: {}; Current Shape: {}'.format(shape_fmt, img.get_data().shape))

        # Do Anything
        #model_file = config.get_path()
        logger.info('Available Model File (you can do something of your choice): {}'.format(model_file))
        
        image = img.get_data()
        output = image.copy()
        #image = np.resize(l,(256,256,37))
        logger.info('Image: {}; Inverted: {}'.format(image.shape,image))
        image =image/ 4095
        image.squeeze()
        image = torch.FloatTensor(image)[None]
      
        normalize = transforms.Normalize((0.5,), (0.5,))
        image = normalize(image)

        logger.info('Image: {}; Inverted: {}'.format(image.shape,image.shape))
       
        # Get your own network here
        # assume you put your PyTorch model code in network.py
        # and the model class is called CustomModel
        EXP_GROUPS = {}


        EXP_GROUPS["open_source_unet2d"] = hu.cartesian_exp_group({
            'batch_size': 1,
            'num_channels':1,
            'dataset': [{
                'name':'open_source', 
                'transform':'basic', 
                'transform_mode':3
            }],
            'dataset_size':[{
                'train':'all', 
                'val':'all'
            }],
            'max_epoch': [100],
            'optimizer': ["adam"], 
            'lr': [1e-5,],
            'model': [{
                'name':'semseg', 
                'base':'unet2d', 
                'loss':'dice', 
                'pretrained':'checkpoints/unet2d_aio_pre_luna.ckpt'
            }],
        })
        exp_list = []
        exp_list += EXP_GROUPS['open_source_unet2d']
        for exp_dict in exp_list:
            network = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                                 train_set=None)
      
        h, w, c = image.shape[1:4]
        
        for i in range(c):
            im = image[:,:,:,i:i+1]
            
            output[:,:,i] = network.vis_on_batch(im.permute(3,0,1,2))
       
        transform_ctx.set_image('model', MedicalImage(output, shape_fmt))

        return transform_ctx
