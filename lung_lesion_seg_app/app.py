# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
from os.path import splitext, basename, join
from shutil import copyfile
import tempfile

import json
import logging
import logging.config
import time
import SimpleITK as sitk


from clara.logging import perf_logger

from tensorrtserver.api import ProtocolType
from tensorrtserver.api import ServerHealthContext

from ai4med.workflows.evaluators.bulk_evaluator import BulkEvaluator
#from executor import Executor
from writers.mhd_writer import *
from writers.classification_result_writer import ClassificationResultWriter
from utils.ai4med_components import ComponentBuilder
from inferers.trtis_simple_inferer import TRTISSimpleInferer
from inferers.trtis_sw_inferer import TRTISScanWindowInferer
from custom_inference.custom_inference import CustomInference

class App(object):
    """Executes data transforms and inference on TRTIS.

    This class loads transform configuration, and performs inference on TRTIS.
    It uses Clara Train components as well as the inference/validation config,
    though limited to transforms, TRTIS model loader, and TRTIS Scan Window
    inferer. The Writers are predefined by this application.
    """

    SUPPORTED_EXTENSIONS = ('.nii.gz', '.nii', '.mhd', '.png')
    INPUT_APPLIED_KEY = 'image'      # The key used for input data element.
    OUTPUT_APPLIED_KEY = 'model'     # The key used for output data element.
    OUTPUT_DEFAULT_DTYPE = "uint8"   # The default data type for output image data
    RENDERSERVER_CONFIG_EMBEDED = 'app_base_inference_v2/config/config_render.json'
    RENDERSERVER_META_FILE_NAME = 'config.meta'


    def __init__(self, runtime_env=None):
        """Initialize the App object with required parameter.

        Args:
            config_path (str): The path to the inference config file, defined by Clara Train TLT.
            runtime_env (RuntimeEnv): Object containing runtime settings, if not defaults.
            trtis_uri (str): The TRTIS URL string.
        """
        perf_logger.stage_started('Application setup', 'AI-base_inference Application setup')

        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.runtime_env = runtime_env if runtime_env is not None else RuntimeEnv()
        self.logger.info('Runtime env settings: %s', vars(self.runtime_env))

        # Model name will be retrieved later from the model config file
        self.model_name = None
        self.config_path = self.runtime_env.config_inference_path

        self.trtis_uri = self.runtime_env.trtis_uri
        uri_fragments = self.trtis_uri.split(':')
        self.nii_extension = self.runtime_env.nii_extension
        self.trtis_host = uri_fragments[0]
        self.trtis_port = uri_fragments[1]
        self.inference_context = None

        self.inference_config = None
        self.writers_config = None
        # Limit batch_size to 1, as this refers to the number of input image files
        self.batch_size = 1
        self.pre_transforms = []
        self.post_transforms = []
        self.writers = []
        self.inferer = None
        self.model_loader = None
        self.component_builder = ComponentBuilder()

        # Add a tempdir, to be deleted and re-created after execute.
        self.temp_folder = tempfile.mkdtemp()

        # Now set up all the required objects.
        self.setup()

        perf_logger.stage_ended('Application setup', 'AI-base_inference Application setup')

    def setup(self):
        """This method sets up all required objects before processing input.

        This is an early initialization phase where the app can do any
        required early-stage setup or initialization,

        For this specific application, the static configuration of the app
        can be loaded to initialize the app runtime settings, though the input
        data and output destination should be available at this point.
        """

        # Retrieve from inference/validation config settings in order to create
        # the processing objects, e.g. transforms, model loader, inferer, etc.
        try:
            with open(self.config_path) as fp:
                self.inference_config = json.load(open(self.config_path))
        except:
            self.logger.exception("Failed to load config file: {}".format(self.config_path))
            raise

        pre_transform_config = self.inference_config.get('pre_transforms', [])
        post_transform_config = self.inference_config.get('post_transforms', [])
        self.writers_config = self.inference_config.get('writers', None)
        inferer_config = self.inference_config.get('inferer', None)
        model_loader_config = self.inference_config.get('model_loader', None)

        #TODO Validate the required config, e.g. inferer

        # setup pre and post-processing, if any
        if pre_transform_config:
            self.pre_transforms = [self.build_component(t) for t in pre_transform_config]
            self.pre_transforms = list(filter(None.__ne__, self.pre_transforms))

        if post_transform_config:
            self.post_transforms = [self.build_component(t) for t in post_transform_config]
            self.post_transforms = list(filter(None.__ne__, self.post_transforms))

        # Set up the inferer. For this app, only TRTIS infers are allowed
        if inferer_config:
            # the TRTIS simple and scanning window inferer have different init param names
            if inferer_config['args'].get('ip'):
                inferer_config['args']['ip'] = self.trtis_host
            else:
                inferer_config['args']['server_ip'] = self.trtis_host

            inferer_config['args']['port'] = self.trtis_port

            # Get the model name from the inferer config args
            self.model_name = inferer_config['args'].get('model_name', '')
            if not self.model_name:
                raise Exception('Failed to retrieve the AI model name from inferer config.')
            self.logger.info('Model name:{}'.format(self.model_name))

            protocol = inferer_config['args'].get('protocol', 'HTTP')
            output_type = inferer_config['args'].get('output_type', 'RAW')

            # check model health and wait until the model is ready
            self.check_model_health()

            inferer_name = inferer_config.get('name', '')
            # TODO: Actually can provide the path and use the component_builder
            if inferer_name.lower().endswith('simpleinferer'):
                self.inferer = TRTISSimpleInferer(model_name=self.model_name,
                                                  ip=self.trtis_host,
                                                  port=self.trtis_port,
                                                  protocol=protocol,
                                                  output_type=output_type) 
            else:    
                # self.inferer_1 = TRTISScanWindowInferer(model_name=self.model_name,
                #                                       ip=self.trtis_host,
                #                                       port=self.trtis_port,
                #                                       protocol=protocol,
                #                                       output_type=output_type)
                self.inferer = CustomInference

        # Set up the model loader. For this app, TRTIS loader is needed.
        # model_loader_config:
        #     self.logger.warning('model_loader config item is not used.')
        # self.model_loader = self.inferer_1.model_loader

        # if
        # Delay writers setup as the output file path depends on the input file name.

    def select_input_file(self, input_folder, extensions=SUPPORTED_EXTENSIONS):
        """Returned the selected files path and extension.

        :argument:
            input_folder (string): the path of the folder containing the input file(s)
            extensions (array): the supported file formats identified by the extensions.
        :returns
            file_path (string) : the path of the selected file
            ext (string): the extension of the selected file
        """
        if os.path.isdir(input_folder):
            for file_name in os.listdir(input_folder):
                file_path = os.path.join(input_folder, file_name)
                # if os.path.isfile(file_path):
                #     for ext in extensions:
                #         if file_path.endswith(ext):
                return file_path, '.nii'
            # raise IOError('No supported input file found ({})'.format(extensions))
        elif os.path.isfile(input_folder):
            return input_folder, None
        else:
            raise FileNotFoundError('Argument "input_folder" is not found.')

    def check_model_health(self):
        protocol = ProtocolType.from_str(self.inference_config['inferer']['args']['protocol'])

        self.max_retry = 2 * 60 * 10 # about 10 minutes
        trial = 0
        while True:
            try:
                trial += 1
                health_context = ServerHealthContext(self.trtis_uri, protocol)
                self.logger.info("Trying to check TRTIS server health ...")
                if health_context.is_ready():
                    break
                raise
            except Exception as ex:
                if trial >= self.max_retry:
                    self.logger.exception('Failed to get server status: %s', ex)
                    raise
                else:
                    time.sleep(0.5)

    def execute(self, runtime_env=None):
        """Perform inference with the payload from the Clara pipeline

        args:
            runtime_env (runtime_env): object containing runtime settings

        returns:
            None
        """

        perf_logger.stage_started('Application execute', 'AI-base_inference Application execute')
        perf_logger.stage_started('Input selection and conversion', 'AI-base_inference Input selection and conversion')

        if not runtime_env:
            runtime_env = self.runtime_env

        # Expect only one input folder here.
        input_folder = runtime_env.input_folder
        input_path, input_file_ext = self.select_input_file(input_folder)
        self.logger.info('Selected input file: {}, and its extension: {}'.format(input_path, input_file_ext))

        # create list of file paths/labels as required by the executor, nii format.
        # Converting .mhd to .nii is needed due to limitation in pre-transforms
        input_path_for_transform = input_path

        # Init the image meta data dictionary, for MHD support only
        mhd_property_dict = None

        # Convert mhd input to nifti as the pre-transform expects nifti file (at least for now).
        if input_file_ext.casefold() == '.mhd'.casefold():
            nii_extension = self.nii_extension
            input_path_for_transform, mhd_property_dict = self._mhd_to_nii(input_path, nii_extension)
            self.logger.info('Input file converted for transforms: %s. Extracted properties: %s.',
                input_path_for_transform, mhd_property_dict)
        file_list = [{App.INPUT_APPLIED_KEY: input_path_for_transform}]
        self.logger.info('File list for transforms: %s', file_list)

        # Expect only one output folder here.
        output_folder = runtime_env.output_folder

        # The output file name is same with the input file name (but with .mhd extension)
        # due to the current limitation of Dicom writer.
        # TODO: CVI-1450 - Improve DICOM writer operator to add support for Nifti format
        if input_file_ext == '.png':
            threshold = self.writers_config[0].get('threshold', 0.5)
            output_file_extension = '.png'
            output_path = os.path.join(output_folder, 'output-' + basename(
                input_path).replace(input_file_ext, output_file_extension))
            self.writers = [ClassificationResultWriter(
                App.OUTPUT_APPLIED_KEY, output_path, input_path, threshold)]
        else:
            output_file_extension = '.mhd'
            output_path = os.path.join(output_folder,
                                       basename(input_path).replace(input_file_ext, output_file_extension))
            # setup writers that will write to disk any of the results
            self.writers = self._build_output_writer(
                output_path, mhd_property_dict)

        perf_logger.stage_ended('Input selection and conversion', 'AI-base_inference Input selection and conversion')
        self.logger.info('Generated Output file name: %s', output_path)

        # Use ai4med evaluator
        # inference_executor = BulkEvaluator(
        #     data_dict_list= file_list,
        #     data_prop=None,
        #     model_loader=self.model_loader,
        #     inferer=self.inferer,
        #     batch_size=1,
        #     pre_transforms=self.pre_transforms,
        #     post_transforms=self.post_transforms,
        #     output_writers=self.writers,
        #     label_transforms=None,
        #     val_metrics=None,
        #     do_validation=False,
        #     output_infer_result=True,
        #     overwrite_previous_result=True)

        perf_logger.stage_started('Transforms and Inference', 'AI-base_inference Transforms and Inference')

        # try:
        #     inference_executor.evaluate()
        # except Exception as ex:
        #     self.logger.exception('Log and propogate exception: {}'.format(ex))
        #     raise
        # finally:
        #     inference_executor.close()

        perf_logger.stage_ended('Transforms and Inference', 'AI-base_inference Transforms and Inference')

        # Publish study data if publish folder exists.
        if runtime_env.publish_folder and input_file_ext not in ('.png',):
            # Provide tuple of (file path for input image in .nii or .mhd, publish file name in .nii or .mhd).
            # Due to limitation of the downstream DICOM writer, the output file name needs to be the same as input,
            # so when publishing both input and output image files to the same folder, need to modify the output
            # file name.
            publish_output_path = output_path.replace(output_file_extension, '.output{}'.format(output_file_extension))
            self.publish_data((input_path, os.path.join(runtime_env.publish_folder, os.path.basename(input_path))),
                              (output_path, os.path.join(runtime_env.publish_folder, os.path.basename(publish_output_path))),
                              (App.RENDERSERVER_CONFIG_EMBEDED,
                                   join(runtime_env.publish_folder, 'config_render.json')))

        # Clean up tempdir and recreate it
        shutil.rmtree(self.temp_folder)
        self.temp_folder = tempfile.mkdtemp()
        perf_logger.stage_ended('Application execute', 'AI-base_inference Application execute')

    def _build_output_writer(self, output_path, _=None):
        """Build own output writer for mhd.

        This custom output writer is needed as the current TLT transform lib
        does not support mhd image file loading or writing. Since we are
        minimizing changes needed for transitoning from TLT to deploy, the TLT
        config for writer will be parsed, and the datatype is used to for the
        mhd output.

        Args:
            output_path (str): The path for saving output image data
            property_dict (dict): Deprecated and not needed. Dictionary of properties to set in the new image

        Returns:
            list (Writer): A list of writers.
        """

        if not output_path or len(output_path) < 0:
            raise ValueError('Argument "output_path" cannot be null or empty.')

        # setup writers that will write to disk any of the results
        output_writers = []
        dtype_str = App.OUTPUT_DEFAULT_DTYPE

        if self.writers_config:
            self.logger.warning('Writers in config ignored; pre-defined writer is used.')

        # Try to reuse the datatype from the writer config
        try:
            for t in self.writers_config:
                dtype_str = t["args"]["dtype"]
                break
        except Exception:
            self.logger.warning('Default output data type "%s" will be used instead.', dtype_str)

        output_writers.append(
            MhdWriter(field=App.OUTPUT_APPLIED_KEY,
                      write_path=output_path,
                      compressed=False,
                      dtype=dtype_str
                     )
        )

        return output_writers

    def copy_image_file(self, src, dest):
        self.logger.info('Copying image files "{}" to "{}"'.format(src, dest))
        src_name, _ = splitext(src)
        dest_name, _ = splitext(dest)
        copyfile(src, dest)

        # If the src extension is '.mhd', then copy over the raw file, whose extension could be '.raw' or '.zraw'
        # For now, only do case sensitive match of 'mhd', and use raw file extension as is.
        if src.lower().endswith('.mhd'):
            raw_file_ext = self.update_mhd_metadata(dest)
            src_raw_file = '{}{}'.format(src_name, raw_file_ext)
            dest_raw_file = '{}{}'.format(dest_name, raw_file_ext)
            self.logger.info('Copying source raw file {} to destination {}'.format(src_raw_file, dest_raw_file))
            copyfile(src_raw_file, dest_raw_file)

    def update_mhd_metadata(self, file_name):
        """Update metadata to indicate the raw file name correctly

        Args:
            file_name (string): path of the mhd file

        Returns:
            string: file extension of the original raw file including the '.'.
        """

        file_name_only = basename(file_name)
        lines = []
        raw_file_ext = '.raw'
        with open(file_name, encoding="utf-8") as fp:
            for line in fp:
                line_items = line.split()
                if len(line_items) == 3 and line_items[0].casefold() == 'ElementDataFile'.casefold():
                    name, _ = os.path.splitext(file_name_only)
                    raw_file_ext = os.path.splitext(line_items[2])[1]
                    line_items[2] = "{}{}".format(name, raw_file_ext)
                    line = " ".join(line_items)
                lines.append(line)
        with open(file_name, 'w', encoding="utf-8") as fp:
            fp.write("".join(lines))

        return raw_file_ext

    def publish_data(self, orig_volume, label_volume, render_config):
        """Publish original volume, labeled volume and render server configuration files
        """
        try:
            self.copy_image_file(orig_volume[0], orig_volume[1])
            self.copy_image_file(label_volume[0], label_volume[1])
            copyfile(render_config[0], render_config[1])

            # Generate meta data describing the model name and original/segmented volumes files
            metadata_published_content = {
                'name': self.model_name,
                'density': os.path.basename(orig_volume[1]),
                'settings': os.path.basename(render_config[1]),
                'seg_masks': [os.path.basename(label_volume[1])]
            }

            # Save metadata in the same dir as the other files.
            with open(os.path.join(os.path.dirname(render_config[1]), App.RENDERSERVER_META_FILE_NAME), 'w') as outfile:
                json.dump(metadata_published_content, outfile, indent=4, sort_keys=True)

        except Exception as ex:
            self.logger.warning('Failed to publish data because: %s', ex)

    def build_component(self, config_dict):
        """Method for building transform instance from config dictionary

        Args:
            config_dict: A dictionary that contains the name of the transform and constructor input
            arguments.
        """

        self.logger.info('Config details: %s', config_dict)
        return self.component_builder.build_component(config_dict)

    # Need to use this method to convert mhd to nii before the mhd support is in transforms lid.
    def _mhd_to_nii(self, mhd_filepath, nii_extension):
        """Convert tge MHD file to nifti format.

        Args:
            mhd_filepath (str): Path of the input mhd file
            nii_extension (str): Extension of the nii file. (compressed or uncompressed)

        Return:
            str, dict: The path to nifti file, image property dict {Spacing, Origin, Direction}.
            """
        # Sanity check
        if not os.path.isfile(mhd_filepath) or not mhd_filepath.endswith('.mhd'):
            raise IOError('Input file is not mhd: {}'.format(mhd_filepath))

        # write out the converted file to the temp dir
        new_input_path = os.path.join(self.temp_folder,
                                      os.path.basename(mhd_filepath).replace('.mhd', nii_extension))
        # Read the mdh image and write out in nii image
        image = sitk.ReadImage(mhd_filepath)
        sitk.WriteImage(image, new_input_path)

        # Capture the original mhd image key properties
        property_dict = {'Spacing': None,
                         'Origin': None,
                         'Direction': None
                         }
        try:
            property_dict['Spacing'] = image.GetSpacing()
        except Exception as ex:
            logging.error('Best effort to get image Spacing failed: %s', ex)
            pass

        try:
            property_dict['Origin'] = image.GetOrigin()
        except Exception as ex:
            logging.error('Best effort to get image origin failed: %s', ex)
            pass

        try:
            property_dict['Direction'] = image.GetDirection()
        except Exception as ex:
            logging.error('Best effort to get image Direction failed: %s', ex)
            pass

        # Try to parse out the metadata
        for key in image.GetMetaDataKeys():
            try:
                logging.info('MHD metadata[%s]: %s', key, image.GetMetaData(key))
            except Exception:
                # Best effort to extract each key value
                pass

        return new_input_path, property_dict


class RuntimeEnv(object):
    """Class responsible to managing run time settings.

    The expected environment variables are the keys in the defaults dictionary,
    and they can be set to override the defaults.
    """

    ENV_DEFAULT = {
        'NVIDIA_CLARA_INPUT': '/input',
        'NVIDIA_CLARA_OUTPUT': '/output',
        'NVIDIA_CLARA_LOGS': '/logs',
        'NVIDIA_CLARA_PUBLISHING': '/publish',
        'NVIDIA_CLARA_TRTISURI': 'localhost:8000',
        'NVIDIA_CLARA_CONFIG_INFERENCE': 'app_base_inference_v2/config/config_inference.json',
        'NVIDIA_CLARA_NII_EXTENSION': '.nii'
    }

    def __init__(self):
        self.input_folder = os.environ.get('NVIDIA_CLARA_INPUT',
                                           RuntimeEnv.ENV_DEFAULT['NVIDIA_CLARA_INPUT'])
        self.output_folder = os.environ.get('NVIDIA_CLARA_OUTPUT',
                                            RuntimeEnv.ENV_DEFAULT['NVIDIA_CLARA_OUTPUT'])
        self.logs_folder = os.environ.get('NVIDIA_CLARA_LOGS',RuntimeEnv.ENV_DEFAULT['NVIDIA_CLARA_LOGS'])
        self.publish_folder = os.environ.get('NVIDIA_CLARA_PUBLISHING',
                                             RuntimeEnv.ENV_DEFAULT['NVIDIA_CLARA_PUBLISHING'])
        self.trtis_uri = os.environ.get('NVIDIA_CLARA_TRTISURI',RuntimeEnv.ENV_DEFAULT['NVIDIA_CLARA_TRTISURI'])
        self.config_inference_path = os.environ.get('NVIDIA_CLARA_CONFIG_INFERENCE',
                                                    RuntimeEnv.ENV_DEFAULT['NVIDIA_CLARA_CONFIG_INFERENCE'])
        self.nii_extension = os.environ.get('NVIDIA_CLARA_NII_EXTENSION',
                                             RuntimeEnv.ENV_DEFAULT['NVIDIA_CLARA_NII_EXTENSION'])

