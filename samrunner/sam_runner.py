# Copyright 2023 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: BSD-3

import numpy as np
from base_runner import BaseRunner
from base_runner import Feature
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import csv

class SAMRunner(BaseRunner):
    """
    SAM agent to continuously monitor given folders and write out segmentations
    """

    def __init__(self, input_dir, output_folder, trigger_file, model_type, checkpoint_dir, device):
        super().__init__(input_dir, output_folder, trigger_file, model_type, device)
        checkpoint_file = self.download_model(self.model_type, Path(checkpoint_dir))
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint_file)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    @staticmethod
    def download_model(model_type: str, target_dir: Path, force=False) -> Path:
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
        return BaseRunner.download_model(url, target_dir, force)

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
        feature_object = Feature(self.predictor.input_size, self.predictor.original_size, self.predictor.features)
        return feature_object

    def get_prompts_from_trigger_file(self):
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

    def run_inference(self, features: Feature, prompts):
        self.set_features_to_predictor(features)
        input_points, input_labels = prompts
        mask, _, _ = self.predictor.predict(point_coords=input_points, point_labels=input_labels,
                                            multimask_output=False)
        return mask
