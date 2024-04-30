# Copyright 2023 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: BSD-3

import os
import sys
import glob
import SimpleITK as sitk
import cv2
import time
import torch
from tqdm import tqdm
from typing import Union, Dict
from pathlib import Path
import numpy as np
import requests
from requests import HTTPError
import hashlib

MITK_META_JSON = u'[{"labels": [{"color": {"type": "ColorProperty","value": [1.0, 0.0, 0.0]},"locked": true,"name": "Label 1","opacity": 1,"value": 1,"visible": true}]}]'


class Feature:
    """
    Class object to store embedding features and its metadata.
    """

    def __init__(self, input_size: tuple = None, original_size: tuple = None, feature_space: np.ndarray = None):
        self.input_size = input_size
        self.original_size = original_size
        self.feature_embeddings = feature_space


class BaseRunner:
    def __init__(self, input_dir, output_folder, trigger_file, model_type, device):
        self.input_dir: Path = Path(input_dir)
        self.output_folder: Path = Path(output_folder)
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.MASTER_RECORD: Dict[str, Feature] = {}
        self.active_file_name: str = None
        self.stop = False
        self.RETRY_LOADING = 10
        self.trigger_file = os.path.join(input_dir, trigger_file)
        self.control_file = os.path.join(input_dir, 'control.txt')
        self.predictor = None

    @staticmethod
    def send_signal(signal: str):
        '''
        Sends signal to caller. For MITK, a simple print should suffice.
        @param signal:
        @return:
        '''
        print(signal)

    def get_features(self, image: Union[np.ndarray, torch.Tensor]) -> Feature:
        '''
        Runs SAM on the input numpy array image. If input is 2D single channel, it's made channel by duplicating.
        @param image:
        @return:
        '''
        pass

    def IsStop(self):
        '''
        Getter for self.stop variable
        @return: bool
        '''
        if not self.stop:
            self.check_control_file()
        return self.stop

    def check_control_file(self):
        '''
        Opens control file and checks for KILL signal in it.
        @return:
        '''
        try:
            with open(self.control_file, mode='r') as file:
                for line in file:
                    if line.upper() == "KILL":
                        self.stop = True
                        self.send_signal('KILL')  # print('KILL')
                        break
                else:
                    self.stop = False
        except IOError:
            self.stop = False

    def get_image_from_file(self, file):
        '''
        Loads image file as numpy array. Retries RETRY_LOADING times with time delay.
        @param file:
        @return: numpy array
        '''
        n_try = 0
        while n_try < self.RETRY_LOADING:
            try:
                data_itk = sitk.ReadImage(file)
                image_2d = sitk.GetArrayFromImage(data_itk)
                image_2d = cv2.normalize(image_2d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image_2d = image_2d.astype(np.uint8, copy=False).squeeze()
                if len(image_2d.shape) == 2:
                    image_3c = np.dstack([image_2d[:, :, None]] * 3)
                else:
                    image_3c = image_2d
            except:
                print('Exception occured, trying again...')
                n_try += 1
                time.sleep(0.1 * n_try)
            else:
                break
        return image_3c

    def get_prompts_from_trigger_file(self):
        '''
        Reads trigger file and parses input points and labels to form prompts for SAM.
        @return:
        '''
        pass

    def set_features_to_predictor(self, features: Feature):
        '''
        Copies SAM predicted embeddings to feature objects.
        @param features:
        @return:
        '''
        self.predictor.features = features.feature_embeddings
        self.predictor.original_size = features.original_size
        self.predictor.input_size = features.input_size
        self.predictor.is_image_set = True

    def start_agent(self):
        pass

    @staticmethod
    def download_model(url: str, target_dir: Path, force=False) -> Path:
        file_name = url.split('/')[-1]
        # _cwd = Path(__file__)
        # file_path = _cwd.cwd().parents[1] / file_name  # Go 2 levels up
        file_path = target_dir / file_name
        if file_path.exists() and not force:
            print('Model checkpoint detected.')
            return file_path
        try:
            with requests.get(url, stream=True) as r:
                file_size = int(r.headers.get('Content-Length', 0))
                r.raise_for_status()
                progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, file=sys.stdout)
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                progress_bar.close()
        except HTTPError as http_error:
            print(f'HTTP error occurred: {http_error}')
            raise http_error
        except Exception as error:
            print(f'Error occurred during download: {error}')
            raise error
        return file_path
    
    @staticmethod
    def is_md5sum(filepath, target_m5sum):
        with open(filepath, 'rb') as file:
            data = file.read()    
            md5_sum = hashlib.md5(data).hexdigest()
        print('Md5 sum:', md5_sum)
        return md5_sum == target_m5sum
    
    @staticmethod
    def write_image_to_disk(mask: np.ndarray, output_path: str):
        seg_image_itk = sitk.GetImageFromArray(mask.astype(np.uint8, copy=False))
        seg_image_itk.SetMetaData('modality', u'org.mitk.multilabel.segmentation')
        seg_image_itk.SetMetaData('org.mitk.multilabel.segmentation.labelgroups', MITK_META_JSON)
        seg_image_itk.SetMetaData('org.mitk.multilabel.segmentation.unlabeledlabellock', '0')
        seg_image_itk.SetMetaData('org.mitk.multilabel.segmentation.version', '1')
        sitk.WriteImage(seg_image_itk, output_path)

    def pre_process_image(self, image):
        return image

    def run_inference(self, features: Feature, prompts):
        pass

    def start_agent(self):
        '''
        The agent which coordinates all the operations of monitoring, predicting and caching SAM inferencing.
        @return:
        '''
        path_template = os.path.join(self.input_dir, '*.nrrd')
        self.send_signal('READY')
        while not glob.glob(path_template):
            time.sleep(0.1)  # wait until image file is found in the input folder
            if self.IsStop(): break
        while True:  # Main Daemon while loop
            if self.IsStop(): break
            file_path = Path(glob.glob(path_template)[0])
            print('File found:', file_path)
            self.active_file_name = file_path.name
            if self.IsStop(): break
            if self.active_file_name not in self.MASTER_RECORD:
                print('File NOT found in MASTER RECORD:', self.active_file_name)
                image_3c = self.get_image_from_file(file_path)
                feature_object = self.get_features(image_3c)
                self.MASTER_RECORD[self.active_file_name] = feature_object
            else:
                print('File found in MASTER RECORD:', file_path)
                feature_object = self.MASTER_RECORD[self.active_file_name]
            try:
                os.remove(file_path)
            except:
                print('Delete failed')
            _cached_stamp = 0
            while not glob.glob(self.trigger_file):
                time.sleep(0.01)  # wait until trigger file is found in the input folder
                if self.IsStop(): break
            while True:  # Loop to monitor changes in trigger file
                stamp = os.stat(self.trigger_file).st_mtime
                if stamp != _cached_stamp:
                    prompts = self.get_prompts_from_trigger_file()
                    if self.IsStop():
                        break
                    output_path = os.path.join(self.output_folder, self.active_file_name)
                    try:
                        os.remove(output_path)
                    except:
                        print('Delete failed')
                    mask = self.run_inference(feature_object, prompts)
                    self.write_image_to_disk(mask, output_path)
                    _cached_stamp = stamp
                    print('SUCCESS')
                if self.IsStop() or glob.glob(path_template): break
            if self.IsStop(): break
        print('Agent has stopped...')