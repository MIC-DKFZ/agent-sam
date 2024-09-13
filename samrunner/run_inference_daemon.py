# Copyright 2023 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: BSD-3

import sys
import time
import torch
import argparse
from base_runner import BaseRunner
from sam_runner import SAMRunner
from medsam_runner import MedSAMRunner
import warnings
warnings.filterwarnings("ignore")

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
parser.add_argument("--backend", type=str, default="SAM", help="SAM or MedSAM")

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
        time.sleep(1)  # arbitrary wait time for stdout & stderr pipes to get established into MITK
        if args.backend == 'SAM':
            runner = SAMRunner(args.input_folder, args.output_folder, args.trigger_file, args.model_type,
                               args.checkpoint, args.device)
        else:
            runner = MedSAMRunner(args.input_folder, args.output_folder, args.trigger_file, args.model_type,
                                  args.checkpoint, args.device)
        runner.start_agent()
    except torch.cuda.OutOfMemoryError as e:
        time.sleep(0.5)
        BaseRunner.send_signal('CudaOutOfMemoryError')
        torch.cuda.empty_cache()
        print(e, file=sys.stderr) # Force to stderr
    except Exception as e:
        time.sleep(0.5)
        BaseRunner.send_signal('KILL')
        print(e, file=sys.stderr) # Force to stderr
    time.sleep(0.5) # arbitrary wait time for MITK
    print('Stopping daemon...')
