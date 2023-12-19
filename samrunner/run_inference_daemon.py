# SPDX-FileCopyrightText: Copyright 2023 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: BSD-3

import time
from typing import Dict
import csv
import torch
import os
import argparse
import requests
from pathlib import Path
from requests import HTTPError
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import glob
import SimpleITK as sitk
import cv2
from tqdm import tqdm

MITK_META_JSON = u'[{"labels": [{"color": {"type": "ColorProperty","value": [1.0, 0.0, 0.0]},"locked": true,"name": "Label 1","opacity": 1,"value": 1,"visible": true}]}]'


class Feature:
    """
    Class object to store embedding features and its metadata.
    """
    def __init__(self, input_size: tuple = None, original_size: tuple = None, feature_space: np.ndarray = None):
        self.input_size = input_size
        self.original_size = original_size
        self.feature_embeddings = feature_space


class SAMRunner:
    """
    SAM agent to continuously monitor given folders and write out segmentations
    """
    def __init__(self, input_dir, output_folder, trigger_file, model_type, checkpoint_dir, device):
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
        checkpoint_file = self.download_model(self.model_type, Path(checkpoint_dir))
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint_file)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    @staticmethod
    def send_signal(signal:str):
        '''
        Sends signal to caller. For MITK, a simple print should suffice.
        @param signal:
        @return:
        '''
        print(signal)

    @staticmethod
    def download_model(model_type:str, target_dir: Path, force=False) -> Path:
        '''
        Downloads model checkpoint from internet to the target directory provided.
        @param model_type:
        @param target_dir:
        @param force:
        @return:
        '''
        if model_type == 'vit_h':
            url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
        elif model_type == 'vit_l':
            url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'
        elif model_type == 'vit_b':
            url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
        else:
            raise Exception('Model type not supported')
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
                progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
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
                image_2d = sitk.GetArrayFromImage(data_itk).astype(np.uint8, copy=False).squeeze()
            except:
                print('Exception occured, trying again...')
                n_try += 1
                time.sleep(0.1*n_try)
            else:
                break
        return image_2d

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

    def get_features(self, image: np.ndarray):
        '''
        Runs SAM on the input numpy array image. If input is 2D single channel, it's made channel by duplicating.
        @param image:
        @return:
        '''
        assert image.ndim in (2, 3)
        assert image.dtype == np.uint8
        if image.ndim == 2:
            image = np.dstack([image[:, :, None]] * 3)
        self.predictor.set_image(image)

    def get_points_and_labels_from_trigger_file(self):
        '''
        Reads trigger file and parses input points and labels to form prompts for SAM.
        @return:
        '''
        with open(self.trigger_file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            points = []
            labels = []
            try:
                for row in csv_reader:
                    points.append([int(x) for x in row['Point'].split(' ')])
                    labels.append(int(row['Label']))
            except ValueError:
                self.stop = True
        input_points = np.array(points)
        input_labels = np.array(labels)
        return input_points, input_labels

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
                image_2d = self.get_image_from_file(file_path)
                image_2d = cv2.normalize(image_2d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.get_features(image_2d)
                feature_object = Feature(self.predictor.input_size, self.predictor.original_size,
                                         self.predictor.features)
                self.MASTER_RECORD[self.active_file_name] = feature_object
            else:
                print('File found in MASTER RECORD:', file_path)
                feature_object = self.MASTER_RECORD[self.active_file_name]
                self.set_features_to_predictor(feature_object)
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
                    input_points, input_labels = self.get_points_and_labels_from_trigger_file()
                    print('input points', input_points)
                    print('input labels', input_labels)
                    if self.IsStop():
                        break
                    output_path = os.path.join(self.output_folder, self.active_file_name)
                    try:
                        os.remove(output_path)
                    except:
                        print('Delete failed')
                    mask, _, _ = self.predictor.predict(point_coords=input_points, point_labels=input_labels,
                                                        multimask_output=False)
                    seg_image_itk = sitk.GetImageFromArray(mask.astype(np.uint8, copy=False))
                    seg_image_itk.SetMetaData('modality', u'org.mitk.multilabel.segmentation')
                    seg_image_itk.SetMetaData('org.mitk.multilabel.segmentation.labelgroups', MITK_META_JSON)
                    seg_image_itk.SetMetaData('org.mitk.multilabel.segmentation.unlabeledlabellock', '0')
                    seg_image_itk.SetMetaData('org.mitk.multilabel.segmentation.version', '1')
                    sitk.WriteImage(seg_image_itk, output_path)
                    _cached_stamp = stamp
                    print('SUCCESS')
                if self.IsStop() or glob.glob(path_template): break
            if self.IsStop(): break
        print('SAM agent has stopped...')


parser = argparse.ArgumentParser(description="Runs embedding generation on an input image or directory of images. "
                                             "Requires SimpleITK. Saves resulting embeddings as .pth files.")
parser.add_argument("--input-folder", type=str, required=True,
                    help="Path to folder of NRRD files. Each file is expected to "
                         "be in dim order DxHxW or HxW")
parser.add_argument("--output-folder", type=str, required=True, help="Folder to where masks is be exported.")
parser.add_argument("--trigger-file", type=str, required=True, help="Path to the file where points will be written to.")
parser.add_argument("--model-type", type=str, required=True, help="The type of model to load, in "
                                                                  "['default', 'vit_h', 'vit_l', 'vit_b']")
parser.add_argument("--checkpoint", type=str, required=True, help="The folder path to where SAM checkpoint will be "
                                                                  "found or else will be download.")
parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

args = parser.parse_args()

if __name__ == "__main__":
    start = time.time()
    print('Starting python...')
    args = parser.parse_args()
    print(args.input_folder)
    print(args.output_folder)
    print(args.model_type)
    print(args.device)
    try:
        sam_runner = SAMRunner(args.input_folder, args.output_folder, args.trigger_file, args.model_type,
                               args.checkpoint, args.device)
        sam_runner.start_agent()
    except torch.cuda.OutOfMemoryError as e:
        SAMRunner.send_signal('CudaOutOfMemoryError')
        torch.cuda.empty_cache()
        print(e)
    except Exception as e:
        SAMRunner.send_signal('KILL')
        print(e)
    print('Stopping daemon...')
