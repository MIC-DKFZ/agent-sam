# Copyright 2023 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: BSD-3

import os
import numpy as np
import torch
import csv
from base_runner import BaseRunner
from base_runner import Feature
from segment_anything import sam_model_registry
from pathlib import Path
from skimage import transform
import torch.nn.functional as F


class MedSAMRunner(BaseRunner):
    """
    MedSAM agent to continuously monitor given folders and write out segmentations
    """

    def __init__(self, input_dir, output_folder, trigger_file, model_type, checkpoint_dir, device):
        super().__init__(input_dir, output_folder, trigger_file, model_type, device)
        if self.model_type == 'vit_b':
            checkpoint_file = self.download_model(self.model_type, Path(checkpoint_dir))
            self.predictor = sam_model_registry[self.model_type](checkpoint=checkpoint_file)  # change
            self.predictor.to(device=self.device)
            self.predictor.eval()
        else:
            raise Exception('Model type not supported')

    @staticmethod
    def download_model(model_type: str, target_dir: Path, force=False) -> Path:
        '''
        Downloads model checkpoint from internet to the target directory provided.
        @param model_type:
        @param target_dir:
        @param force:
        @return:
        '''
        if model_type == 'vit_b':
            url = 'https://zenodo.org/records/10689643/files/medsam_vit_b.pth'
        else:
            raise Exception('Model type not supported')
        n_try = 5
        while n_try !=0:
            filepath = BaseRunner.download_model(url, target_dir, force)
            if BaseRunner.is_md5sum(filepath, '3bb6db55bd0c9ca30b61248bca72f8d6'):
                return filepath
            else:
                n_try -= 1
                os.remove(filepath)
                print('Downloaded file is corrupted. Please check your internet connection. Retrying...')

    def get_features(self, image):
        '''
        Runs SAM on the input numpy array image. If input is 2D single channel, it's made channel by duplicating.
        @param image:
        @return:
        '''
        processed_image = self.pre_process_image(image)
        embeddings = self.predictor.image_encoder(processed_image)
        H, W, _ = image.shape
        feature_object = Feature(image.shape[:2], (H, W), embeddings)
        return feature_object

    def pre_process_image(self, image_3c):
        img_1024 = transform.resize(
            image_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
        # convert the shape to (3, H, W)
        img_1024_tensor = (torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device))
        return img_1024_tensor

    def get_prompts_from_trigger_file(self):
        '''
        Reads trigger file and parses coordinates to form prompts for MedSAM.
        @return:
        '''
        with open(self.trigger_file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            try:
                for row in csv_reader:
                    points = [int(x) for x in row['Coordinates'].split(' ')]
                    break  # Read only first line
            except ValueError:
                self.stop = True
        input_points = np.array([points])
        return input_points

    def run_inference(self, features: Feature, prompts):
        H, W = features.original_size
        box_1024 = prompts / np.array([W, H, W, H]) * 1024
        return self.medsam_inference(self.predictor, features.feature_embeddings, box_1024, H, W)

    @staticmethod
    @torch.no_grad()
    def medsam_inference(medsam_model, img_embed, box_1024, H, W):
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)
        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False) #(1, 1, gt.shape)
        low_res_pred = low_res_pred.cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg[0]
