# Part of the code of this file is taken from Ultralytics and Model Optimization (MCT)

# ------------------------------------------------------------------------------
# This file contains code from the Ultralytics repository (YOLOv8)
# Copyright (C) 2024  Ultralytics
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Yolov8n Object Detection Model - PyTorch implementation

This code contains a PyTorch implementation of Yolov8n object detection model, following
https://github.com/ultralytics/ultralytics.

Usage:
  model, cfg_dict = yolov8_pytorch("yolov8n.yaml")
  pretrained_weights = torch.load('/path/to/pretrained/yolov8n.pt')['model'].state_dict()
  model.load_state_dict(pretrained_weights, strict=False)
  model.eval()

Main changes:
  Modify layers to make them more suitable for quantization
  torch.fx compatibility
  Detect head (mainly the box decoding part that was optimized for model quantization)
  Inheritance class from HuggingFace
  Implement box decoding into Detect Layer

Notes and Limitations:
- The model has been tested only with the default settings from Ultralytics, specifically using a 640x640 input resolution and 80 object classes.
- Anchors and strides are hardcoded as constants within the model, meaning they are not included in the weights file from Ultralytics.

The code is organized as follows:
- Classes definitions of Yolov8n building blocks: Conv, Bottleneck, C2f, SPPF, Upsample, Concaat, DFL and Detect
- Detection Model definition: ModelPyTorch
- PostProcessWrapper Wrapping the Yolov8n model with PostProcess layer (Specifically, sony_custom_layers/multiclass_nms)
- A getter function for getting a new instance of the model

For more details on the Yolov8n model, refer to the original repository:
https://github.com/ultralytics/ultralytics

"""
import contextlib
import math
import re
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import Tensor
from huggingface_hub import PyTorchModelHubMixin
import importlib

from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
if importlib.util.find_spec("sony_custom_layers"):
    from sony_custom_layers.pytorch.object_detection.nms import multiclass_nms

from enum import Enum
import numpy as np
from typing import List
import cv2

import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
import random

def coco80_to_coco91(x: np.ndarray) -> np.ndarray:
    """
    Converts COCO 80-class indices to COCO 91-class indices.

    Args:
        x (numpy.ndarray): An array of COCO 80-class indices.

    Returns:
        numpy.ndarray: An array of corresponding COCO 91-class indices.
    """
    coco91Indexs = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
         63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90])
    return coco91Indexs[x.astype(np.int32)]

# COCO evaluation class
class CocoEval:
    def __init__(self, path2json: str, output_resize: Dict = None, task: str = 'Detection'):
        """
        Initialize the CocoEval class.

        Args:
            path2json (str): Path to the COCO JSON file containing ground truth annotations.
            output_resize (Dict): Contains the resize information to map between the model's output and the original
             image dimensions. The dict consists of:
                  {"shape": (height, weight),
                   "aspect_ratio_preservation": bool}
        """
        # Load ground truth annotations
        self.coco_gt = COCO(path2json)

        # A list of reformatted model outputs
        self.all_detections = []

        # Resizing information to map between the model's output and the original image dimensions
        self.output_resize = output_resize if output_resize else {'shape': (1, 1), 'aspect_ratio_preservation': False}

        # Set the task type (Detection/Segmentation/Keypoints)
        self.task = task

    def add_batch_detections(self, outputs: Tuple[List, List, List, List], targets: List[Dict]):
        """
        Add batch detections to the evaluation.

        Args:
            outputs (list): List of model outputs, typically containing bounding boxes, scores, and labels.
            targets (list): List of ground truth annotations for the batch.
        """
        img_ids, _outs = [], []
        orig_img_dims = []
        for idx, t in enumerate(targets):
            if len(t) > 0:
                img_ids.append(t[0]['image_id'])
                orig_img_dims.append(t[0]['orig_img_dims'])
                _outs.append([o[idx] for o in outputs])

        batch_detections = self.format_results(_outs, img_ids, orig_img_dims, self.output_resize)

        self.all_detections.extend(batch_detections)

    def result(self) -> List[float]:
        """
        Calculate and print evaluation results.

        Returns:
            list: COCO evaluation statistics.
        """
        # Initialize COCO evaluation object
        self.coco_dt = self.coco_gt.loadRes(self.all_detections)
        if self.task == 'Detection':
            coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox')
        elif self.task == 'Keypoints':
            coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'keypoints')
        else:
            raise Exception("Unsupported task type of CocoEval")

        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Print mAP results
        print("mAP: {:.4f}".format(coco_eval.stats[0]))

        return coco_eval.stats

    def reset(self):
        """
        Reset the list of detections to prepare for a new evaluation.
        """
        self.all_detections = []

    def format_results(self, outputs: List, img_ids: List, orig_img_dims: List, output_resize: Dict) -> List[Dict]:
        """
        Format model outputs into a list of detection dictionaries.

        Args:
            outputs (list): List of model outputs, typically containing bounding boxes, scores, and labels.
            img_ids (list): List of image IDs corresponding to each output.
            orig_img_dims (list): List of tuples representing the original image dimensions (h, w) for each output.
            output_resize (Dict): Contains the resize information to map between the model's
                     output and the original image dimensions.

        Returns:
            list: A list of detection dictionaries, each containing information about the detected object.
        """
        detections = []
        h_model, w_model = output_resize['shape']
        preserve_aspect_ratio = output_resize['aspect_ratio_preservation']
        normalized_coords = output_resize.get('normalized_coords', True)
        align_center = output_resize.get('align_center', True)

        if self.task == 'Detection':
            # Process model outputs and convert to detection format
            for idx, output in enumerate(outputs):
                image_id = img_ids[idx]
                scores = output[1].numpy().squeeze()  # Extract scores
                labels = (coco80_to_coco91(
                    output[2].numpy())).squeeze()  # Convert COCO 80-class indices to COCO 91-class indices
                boxes = output[0].numpy().squeeze()  # Extract bounding boxes
                boxes = scale_boxes(boxes, orig_img_dims[idx][0], orig_img_dims[idx][1], h_model, w_model,
                                    preserve_aspect_ratio, align_center, normalized_coords)

                for score, label, box in zip(scores, labels, boxes):
                    detection = {
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [box[1], box[0], box[3] - box[1], box[2] - box[0]],
                        "score": score
                    }
                    detections.append(detection)

        elif self.task == 'Keypoints':
            for output, image_id, (w_orig, h_orig) in zip(outputs, img_ids, orig_img_dims):

                bbox, scores, kpts = output

                # Add detection results to predicted_keypoints list
                if kpts.shape[0]:
                    kpts = kpts.reshape(-1, 17, 3)
                    kpts = scale_coords(kpts, h_orig, w_orig, 640, 640, True)
                    for ind, k in enumerate(kpts):
                        detections.append({
                            'category_id': 1,
                            'image_id': image_id,
                            'keypoints': k.reshape(51).tolist(),
                            'score': scores.tolist()[ind] if isinstance(scores.tolist(), list) else scores.tolist()
                        })

        return detections

def load_and_preprocess_image(image_path: str, preprocess: Callable) -> np.ndarray:
    """
    Load and preprocess an image from a given file path.

    Args:
        image_path (str): Path to the image file.
        preprocess (function): Preprocessing function to apply to the loaded image.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    image = cv2.imread(image_path)
    image = preprocess(image)
    return image


class CocoDataset:
    def __init__(self, dataset_folder: str, annotation_file: str, preprocess: Callable):
        """
        A dataset class for handling COCO dataset images and annotations.

        Args:
            dataset_folder (str): The path to the folder containing COCO dataset images.
            annotation_file (str): The path to the COCO annotation file in JSON format.
            preprocess (Callable): A function for preprocessing images.
        """
        self.dataset_folder = dataset_folder
        self.preprocess = preprocess

        # Load COCO annotations from a JSON file (e.g., 'annotations.json')
        with open(annotation_file, 'r') as f:
            self.coco_annotations = json.load(f)

        # Initialize a dictionary to store annotations grouped by image ID
        self.annotations_by_image = {}

        # Iterate through the annotations and group them by image ID
        for annotation in self.coco_annotations['annotations']:
            image_id = annotation['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(annotation)

        # Initialize a list to collect images and annotations for the current batch
        self.total_images = len(self.coco_annotations['images'])

    def __len__(self):
        return self.total_images

    def __getitem__(self, item_index):
        """
        Returns the preprocessed image and its corresponding annotations.

        Args:
            item_index: Index of the item to retrieve.

        Returns:
            Tuple containing the preprocessed image and its annotations.
        """
        image_info = self.coco_annotations['images'][item_index]
        image_id = image_info['id']
        image = load_and_preprocess_image(os.path.join(self.dataset_folder, image_info['file_name']), self.preprocess)
        annotations = self.annotations_by_image.get(image_id, [])
        if len(annotations) > 0:
            annotations[0]['orig_img_dims'] = (image_info['height'], image_info['width'])
        return image, annotations

    def sample(self, batch_size):
        """
        Samples a batch of images and their corresponding annotations from the dataset.

        Returns:
            Tuple containing a batch of preprocessed images and their annotations.
        """
        batch_images = []
        batch_annotations = []

        # Sample random image indexes
        random_idx = random.sample(range(self.total_images), batch_size)

        # Get the corresponding items from dataset
        for idx in random_idx:
            batch_images.append(self[idx][0])
            batch_annotations.append(self[idx][1])

        return np.array(batch_images), batch_annotations


class DataLoader:
    def __init__(self, dataset: List[Tuple], batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.count = 0
        self.inds = list(range(len(dataset)))

    def __iter__(self):
        self.count = 0
        if self.shuffle:
            random.shuffle(self.inds)

        return self

    def __next__(self):
        if self.count >= len(self.dataset):
            raise StopIteration

        batch_images = []
        batch_annotations = []

        while len(batch_images) < self.batch_size and self.count < len(self.dataset):
            index = self.inds[self.count]
            image, annotations = self.dataset[index]
            batch_images.append(image)
            batch_annotations.append(annotations)
            self.count += 1

        return np.array(batch_images), batch_annotations


def coco_dataset_generator(dataset_folder: str, annotation_file: str, preprocess: Callable,
                           batch_size: int = 1) -> Tuple:

    """
    Generator function for loading and preprocessing images and their annotations from a COCO-style dataset.

    Args:
        dataset_folder (str): Path to the dataset folder containing image files.
        annotation_file (str): Path to the COCO-style annotation JSON file.
        preprocess (function): Preprocessing function to apply to each loaded image.
        batch_size (int): The desired batch size.

    Yields:
        Tuple[numpy.ndarray, list]: A tuple containing a batch of images (as a NumPy array) and a list of annotations
        for each image in the batch.
    """
    # Load COCO annotations from a JSON file (e.g., 'annotations.json')
    with open(annotation_file, 'r') as f:
        coco_annotations = json.load(f)

    # Initialize a dictionary to store annotations grouped by image ID
    annotations_by_image = {}

    # Iterate through the annotations and group them by image ID
    for annotation in coco_annotations['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Initialize a list to collect images and annotations for the current batch
    batch_images = []
    batch_annotations = []
    total_images = len(coco_annotations['images'])

    # Iterate through the images and create a list of tuples (image, annotations)
    for image_count, image_info in enumerate(coco_annotations['images']):
        image_id = image_info['id']
        # Load and preprocess the image (you can use your own image loading logic)
        image = load_and_preprocess_image(os.path.join(dataset_folder, image_info['file_name']), preprocess)
        annotations = annotations_by_image.get(image_id, [])
        if len(annotations) > 0:
            annotations[0]['orig_img_dims'] = (image_info['height'], image_info['width'])

            # Add the image and annotations to the current batch
            batch_images.append(image)
            batch_annotations.append(annotations)

            # Check if the current batch is of the desired batch size
            if len(batch_images) == batch_size:
                # Yield the current batch
                yield np.array(batch_images), batch_annotations

                # Reset the batch lists for the next batch
                batch_images = []
                batch_annotations = []

        # After processing all images, yield any remaining images in the last batch
        if len(batch_images) > 0 and (total_images == image_count + 1):
            yield np.array(batch_images), batch_annotations


def model_predict(model: Any,
                  inputs: np.ndarray) -> Tuple[List, List, List, List]:
    """
    Perform inference using the provided model on the given inputs.

    This function serves as the default method for inference if no specific model inference function is provided.

    Args:
        model (Any): The model used for inference.
        inputs (np.ndarray): Input data to perform inference on.

    Returns:
        Tuple[List, List, List, List]: Tuple containing lists of predictions.
    """
    return model(inputs)


def coco_evaluate(model: Any, preprocess: Callable, dataset_folder: str, annotation_file: str, batch_size: int,
                  output_resize: tuple, model_inference: Callable = model_predict, task: str = 'Detection') -> dict:
    """
    Evaluate a model on the COCO dataset.

    Args:
    - model (Any): The model to evaluate.
    - preprocess (Callable): Preprocessing function to be applied to images.
    - dataset_folder (str): Path to the folder containing COCO dataset images.
    - annotation_file (str): Path to the COCO annotation file.
    - batch_size (int): Batch size for evaluation.
    - output_resize (tuple): Tuple representing the output size after resizing.
    - model_inference (Callable): Model inference function. model_predict will be used by default.

    Returns:
    - dict: Evaluation results.

    """
    # Load COCO evaluation set
    coco_dataset = CocoDataset(dataset_folder=dataset_folder,
                               annotation_file=annotation_file,
                               preprocess=preprocess)
    coco_loader = DataLoader(coco_dataset, batch_size)

    # Initialize the evaluation metric object
    coco_metric = CocoEval(annotation_file, output_resize, task)

    # Iterate and the evaluation set
    for batch_idx, (images, targets) in enumerate(coco_loader):

        # Run inference on the batch
        outputs = model_inference(model, images)

        # Add the model outputs to metric object (a dictionary of outputs after postprocess: boxes, scores & classes)
        coco_metric.add_batch_detections(outputs, targets)
        if (batch_idx + 1) % 100 == 0:
            print(f'processed {(batch_idx + 1) * batch_size} images')

    return coco_metric.result()

def masks_to_coco_rle(masks, boxes, image_id, height, width, scores, classes, mask_threshold):
    """
    Converts masks to COCO RLE format and compiles results including bounding boxes and scores.

    Args:
        masks (list of np.ndarray): List of segmentation masks.
        boxes (list of np.ndarray): List of bounding boxes corresponding to the masks.
        image_id (int): Identifier for the image being processed.
        height (int): Height of the image.
        width (int): Width of the image.
        scores (list of float): Confidence scores for each detection.
        classes (list of int): Class IDs for each detection.

    Returns:
        list of dict: Each dictionary contains the image ID, category ID, bounding box,
                      score, and segmentation in RLE format.
    """
    results = []
    for i, (mask, box) in enumerate(zip(masks, boxes)):

        binary_mask = np.asfortranarray((mask > mask_threshold).astype(np.uint8))
        rle = mask_utils.encode(binary_mask)
        rle['counts'] = rle['counts'].decode('ascii')

        x_min, y_min, x_max, y_max = box[1], box[0], box[3], box[2]
        box_width = x_max - x_min
        box_height = y_max - y_min

        adjusted_category_id = coco80_to_coco91(np.array([classes[i]]))[0]
        
        result = {
            "image_id": int(image_id),  # Convert to int if not already
            "category_id": int(adjusted_category_id),  # Ensure type is int
            "bbox": [float(x_min), float(y_min), float(box_width), float(box_height)],
            "score": float(scores[i]),  # Ensure type is float
            "segmentation": rle
        }
        results.append(result)
    return results

def save_results_to_json(results, file_path):
    """
    Saves the results data to a JSON file.

    Args:
        results (list of dict): The results data to be saved.
        file_path (str): The path to the file where the results will be saved.
    """
    with open(file_path, 'w') as f:
        json.dump(results, f)

def evaluate_seg_model(annotation_file, results_file):
    """
    Evaluate the model's segmentation performance using the COCO evaluation metrics.

    This function loads the ground truth annotations and the detection results from specified files,
    filters the annotations to include only those images present in the detection results, and then
    performs the COCO evaluation.

    Args:
        annotation_file (str): The file path for the COCO format ground truth annotations.
        results_file (str): The file path for the detection results in COCO format.

    The function prints out the evaluation summary which includes average precision and recall
    across various IoU thresholds and object categories.
    """
   
    coco_gt = COCO(annotation_file)
    coco_dt = coco_gt.loadRes(results_file)
    
    # Extract image IDs from the results file
    with open(results_file, 'r') as file:
        results_data = json.load(file)
    result_img_ids = {result['image_id'] for result in results_data}
    
    # Filter annotations to include only those images present in the results file
    coco_gt.imgs = {img_id: coco_gt.imgs[img_id] for img_id in result_img_ids if img_id in coco_gt.imgs}
    coco_gt.anns = {ann_id: coco_gt.anns[ann_id] for ann_id in list(coco_gt.anns.keys()) if coco_gt.anns[ann_id]['image_id'] in result_img_ids}
    
    # Evaluate only for the filtered images
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.imgIds = list(result_img_ids)  # Ensure evaluation is only on the filtered image IDs
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def yolov8_preprocess_chw_transpose(x: np.ndarray, img_mean: float = 0.0, img_std: float = 255.0, pad_values: int = 114,
                                    size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess an input image for YOLOv8 model with additional CHW transpose (for PyTorch implementation)

    Args:
        x (np.ndarray): Input image as a NumPy array.
        img_mean (float): Mean value used for normalization. Default is 0.0.
        img_std (float): Standard deviation used for normalization. Default is 255.0.
        pad_values (int): Value used for padding. Default is 114.
        size (Tuple[int, int]): Desired output size (height, width). Default is (640, 640).

    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """

    h, w = x.shape[:2]  # Image size
    hn, wn = size  # Image new size
    r = max(h / hn, w / wn)
    hr, wr = int(np.round(h / r)), int(np.round(w / r))
    pad = (
        (int((hn - hr) / 2), int((hn - hr) / 2 + 0.5)),
        (int((wn - wr) / 2), int((wn - wr) / 2 + 0.5)),
        (0, 0)
    )

    x = np.flip(x, -1)  # Flip image channels
    x = cv2.resize(x, (wr, hr), interpolation=cv2.INTER_AREA)  # Aspect ratio preserving resize
    x = np.pad(x, pad, constant_values=pad_values)  # Padding to the target size
    x = (x - img_mean) / img_std  # Normalization
    x = x.transpose([2, 0, 1])
    return x

class BoxFormat(Enum):
    YMIM_XMIN_YMAX_XMAX = 'ymin_xmin_ymax_xmax'
    XMIM_YMIN_XMAX_YMAX = 'xmin_ymin_xmax_ymax'
    XMIN_YMIN_W_H = 'xmin_ymin_width_height'
    XC_YC_W_H = 'xc_yc_width_height'


def convert_to_ymin_xmin_ymax_xmax_format(boxes, orig_format: BoxFormat):
    """
    changes the box from one format to another (XMIN_YMIN_W_H --> YMIM_XMIN_YMAX_XMAX )
    also support in same format mode (returns the same format)

    :param boxes:
    :param orig_format:
    :return: box in format YMIM_XMIN_YMAX_XMAX
    """
    if len(boxes) == 0:
        return boxes
    elif orig_format == BoxFormat.YMIM_XMIN_YMAX_XMAX:
        return boxes
    elif orig_format == BoxFormat.XMIN_YMIN_W_H:
        boxes[:, 2] += boxes[:, 0]  # convert width to xmax
        boxes[:, 3] += boxes[:, 1]  # convert height to ymax
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XMIM_YMIN_XMAX_YMAX:
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XC_YC_W_H:
        new_boxes = np.copy(boxes)
        new_boxes[:, 0] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
        new_boxes[:, 1] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
        new_boxes[:, 2] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
        new_boxes[:, 3] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
        return new_boxes
    else:
        raise Exception("Unsupported boxes format")

def clip_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Clip bounding boxes to stay within the image boundaries.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        numpy.ndarray: Clipped bounding boxes.
    """
    boxes[..., 0] = np.clip(boxes[..., 0], a_min=0, a_max=h)
    boxes[..., 1] = np.clip(boxes[..., 1], a_min=0, a_max=w)
    boxes[..., 2] = np.clip(boxes[..., 2], a_min=0, a_max=h)
    boxes[..., 3] = np.clip(boxes[..., 3], a_min=0, a_max=w)
    return boxes


def scale_boxes(boxes: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int, preserve_aspect_ratio: bool,
                align_center: bool = True, normalized: bool = True) -> np.ndarray:
    """
    Scale and offset bounding boxes based on model output size and original image size.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
        h_image (int): Original image height.
        w_image (int): Original image width.
        h_model (int): Model output height.
        w_model (int): Model output width.
        preserve_aspect_ratio (bool): Whether to preserve image aspect ratio during scaling
        align_center (bool): Whether to center the bounding boxes after scaling
        normalized (bool): Whether treats bounding box coordinates as normalized (i.e., in the range [0, 1])

    Returns:
        numpy.ndarray: Scaled and offset bounding boxes.
    """
    deltaH, deltaW = 0, 0
    H, W = h_model, w_model
    scale_H, scale_W = h_image / H, w_image / W

    if preserve_aspect_ratio:
        scale_H = scale_W = max(h_image / H, w_image / W)
        H_tag = int(np.round(h_image / scale_H))
        W_tag = int(np.round(w_image / scale_W))
        if align_center:
            deltaH, deltaW = int((H - H_tag) / 2), int((W - W_tag) / 2)

    nh, nw = (H, W) if normalized else (1, 1)

    # Scale and offset boxes
    boxes[..., 0] = (boxes[..., 0] * nh - deltaH) * scale_H
    boxes[..., 1] = (boxes[..., 1] * nw - deltaW) * scale_W
    boxes[..., 2] = (boxes[..., 2] * nh - deltaH) * scale_H
    boxes[..., 3] = (boxes[..., 3] * nw - deltaW) * scale_W

    # Clip boxes
    boxes = clip_boxes(boxes, h_image, w_image)

    return boxes


def scale_coords(kpts: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int, preserve_aspect_ratio: bool) -> np.ndarray:
    """
    Scale and offset keypoints based on model output size and original image size.

    Args:
        kpts (numpy.ndarray): Array of bounding keypoints in format [..., 17, 3]  where the last dim is (x, y, visible).
        h_image (int): Original image height.
        w_image (int): Original image width.
        h_model (int): Model output height.
        w_model (int): Model output width.
        preserve_aspect_ratio (bool): Whether to preserve image aspect ratio during scaling

    Returns:
        numpy.ndarray: Scaled and offset bounding boxes.
    """
    deltaH, deltaW = 0, 0
    H, W = h_model, w_model
    scale_H, scale_W = h_image / H, w_image / W

    if preserve_aspect_ratio:
        scale_H = scale_W = max(h_image / H, w_image / W)
        H_tag = int(np.round(h_image / scale_H))
        W_tag = int(np.round(w_image / scale_W))
        deltaH, deltaW = int((H - H_tag) / 2), int((W - W_tag) / 2)

    # Scale and offset boxes
    kpts[..., 0] = (kpts[..., 0]  - deltaH) * scale_H
    kpts[..., 1] = (kpts[..., 1] - deltaW) * scale_W

    # Clip boxes
    kpts = clip_coords(kpts, h_image, w_image)

    return kpts

def clip_coords(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Clip keypoints to stay within the image boundaries.

    Args:
        kpts (numpy.ndarray): Array of bounding keypoints in format [..., 17, 3]  where the last dim is (x, y, visible).
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        numpy.ndarray: Clipped bounding boxes.
    """
    kpts[..., 0] = np.clip(kpts[..., 0], a_min=0, a_max=h)
    kpts[..., 1] = np.clip(kpts[..., 1], a_min=0, a_max=w)
    return kpts


def nms(dets: np.ndarray, scores: np.ndarray, iou_thres: float = 0.5, max_out_dets: int = 300) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) on detected bounding boxes.

    Args:
        dets (np.ndarray): Array of bounding box coordinates of shape (N, 4) representing [y1, x1, y2, x2].
        scores (np.ndarray): Array of confidence scores associated with each bounding box.
        iou_thres (float, optional): IoU threshold for NMS. Default is 0.5.
        max_out_dets (int, optional): Maximum number of output detections to keep. Default is 300.

    Returns:
        List[int]: List of indices representing the indices of the bounding boxes to keep after NMS.

    """
    y1, x1 = dets[:, 0], dets[:, 1]
    y2, x2 = dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return keep[:max_out_dets]

def combined_nms(batch_boxes, batch_scores, iou_thres: float = 0.5, conf: float = 0.001, max_out_dets: int = 300):

    """
    Performs combined Non-Maximum Suppression (NMS) on batches of bounding boxes and scores.

    Parameters:
    batch_boxes (List[np.ndarray]): A list of arrays, where each array contains bounding boxes for a batch.
    batch_scores (List[np.ndarray]): A list of arrays, where each array contains scores for the corresponding bounding boxes.
    iou_thres (float): Intersection over Union (IoU) threshold for NMS. Defaults to 0.5.
    conf (float): Confidence threshold for filtering boxes. Defaults to 0.001.
    max_out_dets (int): Maximum number of output detections per image. Defaults to 300.

    Returns:
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: A list of tuples for each batch, where each tuple contains:
        - nms_bbox: Array of bounding boxes after NMS.
        - nms_scores: Array of scores after NMS.
        - nms_classes: Array of class IDs after NMS.
    """
    nms_results = []
    for boxes, scores in zip(batch_boxes, batch_scores):

        xc = np.argmax(scores, 1)
        xs = np.amax(scores, 1)
        x = np.concatenate([boxes, np.expand_dims(xs, 1), np.expand_dims(xc, 1)], 1)

        xi = xs > conf
        x = x[xi]

        x = x[np.argsort(-x[:, 4])[:8400]]
        scores = x[:, 4]
        x[..., :4] = convert_to_ymin_xmin_ymax_xmax_format(x[..., :4], BoxFormat.XC_YC_W_H)
        offset = x[:, 5] * 640
        boxes = x[..., :4] + np.expand_dims(offset, 1)

        # Original post-processing part
        valid_indexs = nms(boxes, scores, iou_thres=iou_thres, max_out_dets=max_out_dets)
        x = x[valid_indexs]
        nms_classes = x[:, 5]
        nms_bbox = x[:, :4]
        nms_scores = x[:, 4]

        nms_results.append((nms_bbox, nms_scores, nms_classes))

    return nms_results

def postprocess_yolov8_keypoints(outputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       conf: float = 0.001,
                       iou_thres: float = 0.7,
                       max_out_dets: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess the outputs of a YOLOv8 model for pose estimation.

    Args:
        outputs (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple containing the model outputs for bounding boxes,
            scores and keypoint predictions.
        conf (float, optional): Confidence threshold for bounding box predictions. Default is 0.001.
        iou_thres (float, optional): IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS).
            Default is 0.7.
        max_out_dets (int, optional): Maximum number of output detections to keep after NMS. Default is 300.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the post-processed bounding boxes,
            their corresponding scores and keypoints.
    """
    kpt_shape = (17, 3)
    feat_sizes = np.array([80, 40, 20])
    stride_sizes = np.array([8, 16, 32])
    a, s = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))

    y_bb, y_cls, kpts = outputs
    dbox = dist2bbox_yolo_v8(y_bb, np.expand_dims(a, 0), xywh=True, dim=1) * s
    detect_out = np.concatenate((dbox, y_cls), 1)
    # additional part for pose estimation
    ndim = kpt_shape[1]
    pred_kpt = kpts.copy()
    if ndim == 3:
        pred_kpt[:, 2::3] = 1 / (1 + np.exp(-pred_kpt[:, 2::3]))  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
    pred_kpt[:, 0::ndim] = (pred_kpt[:, 0::ndim] * 2.0 + (a[0] - 0.5)) * s
    pred_kpt[:, 1::ndim] = (pred_kpt[:, 1::ndim] * 2.0 + (a[1] - 0.5)) * s

    x_batch = np.concatenate([detect_out.transpose([0, 2, 1]), pred_kpt.transpose([0, 2, 1])], 2)
    nms_bbox, nms_scores, nms_kpts = [], [], []
    for x in x_batch:
        x = x[(x[:, 4] > conf)]
        x = x[np.argsort(-x[:, 4])[:8400]]
        x[..., :4] = convert_to_ymin_xmin_ymax_xmax_format(x[..., :4], BoxFormat.XC_YC_W_H)
        boxes = x[..., :4]
        scores = x[..., 4]

        # Original post-processing part
        valid_indexs = nms(boxes, scores, iou_thres=iou_thres, max_out_dets=max_out_dets)
        x = x[valid_indexs]
        nms_bbox.append(x[:, :4])
        nms_scores.append(x[:, 4])
        nms_kpts.append(x[:, 5:])

    return nms_bbox, nms_scores, nms_kpts


def postprocess_yolov8_inst_seg(outputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       conf: float = 0.001,
                       iou_thres: float = 0.7,
                       max_out_dets: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    feat_sizes = np.array([80, 40, 20])
    stride_sizes = np.array([8, 16, 32])
    a, s = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))

    y_bb, y_cls, y_masks = outputs
    dbox = dist2bbox_yolo_v8(y_bb, a, xywh=True, dim=1) * s
    detect_out = np.concatenate((dbox, y_cls), 1)

    xd = detect_out.transpose([0, 2, 1])

    return combined_nms(xd[..., :4], xd[..., 4:84], iou_thres, conf, max_out_dets)


def make_anchors_yolo_v8(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i], feats[i]
        sx = np.arange(stop=w) + grid_cell_offset  # shift x
        sy = np.arange(stop=h) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack((sx, sy), -1).reshape((-1, 2)))
        stride_tensor.append(np.full((h * w, 1), stride))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)


def dist2bbox_yolo_v8(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.split(distance,2,axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), dim)  # xyxy bbox

def yaml_load(file: str = 'data.yaml', append_filename: bool = False) -> Dict[str, any]:
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string
        if not s.isprintable():  # remove special characters
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""

        y1 = self.cv1(x).chunk(2, 1)
        y = [y1[0], y1[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    for i, stride in enumerate(strides):
        h, w = int(feats[i]), int(feats[i])
        sx = torch.arange(end=w, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class Detect(nn.Module):
    def __init__(self, nc: int = 80,
                 ch: List[int] = ()):
        """
        Detection layer for YOLOv8.

        Args:
            nc (int): Number of classes.
            ch (List[int]): List of channel values for detection layers.

        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.Tensor([8, 16, 32])
        self.feat_sizes = torch.Tensor([80, 40, 20])
        self.img_size = 640  # img size
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3),
                          Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3),
                                               nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        anchors, strides = (x.transpose(0, 1) for x in make_anchors(self.feat_sizes,
                                                                    self.stride, 0.5))
        anchors = anchors * strides

        self.register_buffer('anchors', anchors)
        self.register_buffer('strides', strides)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.nc), 1)

        y_cls = cls.sigmoid().transpose(1, 2)

        dfl = self.dfl(box)
        dfl = dfl * self.strides

        # box decoding
        lt, rb = dfl.chunk(2, 1)
        y1 = self.anchors.unsqueeze(0)[:, 0, :] - lt[:, 0, :]
        x1 = self.anchors.unsqueeze(0)[:, 1, :] - lt[:, 1, :]
        y2 = self.anchors.unsqueeze(0)[:, 0, :] + rb[:, 0, :]
        x2 = self.anchors.unsqueeze(0)[:, 1, :] + rb[:, 1, :]
        y_bb = torch.stack((x1, y1, x2, y2), 1).transpose(1, 2)
        return y_bb, y_cls

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Detect_wo_bb_dec(nn.Module):
    def __init__(self, nc: int = 80,
                 ch: List[int] = ()):
        """
        Detection layer for YOLOv8. Bounding box decoding was removed.
        Args:
            nc (int): Number of classes.
            ch (List[int]): List of channel values for detection layers.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.Tensor([8, 16, 32])
        self.feat_sizes = torch.Tensor([80, 40, 20])
        self.img_size = 640  # img size
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3),
                          Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3),
                                               nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.nc), 1)

        y_cls = cls.sigmoid()
        y_bb = self.dfl(box)
        return y_bb, y_cls


    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class Pose(Detect_wo_bb_dec):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect_wo_bb_dec.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        y_bb, y_cls = self.detect(self, x)
        return y_bb, y_cls, kpt

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (
                Conv,
                Bottleneck,
                SPPF,
                C2f,
                nn.ConvTranspose2d,
        ):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in [C2f]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in [Segment, Detect, Pose]:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def model_predict(model: Any,
                  inputs: np.ndarray) -> List:
    """
    Perform inference using the provided PyTorch model on the given inputs.

    This function handles moving the inputs to the appropriate torch device and data type,
    and detaches and moves the outputs to the CPU.

    Args:
        model (Any): The PyTorch model used for inference.
        inputs (np.ndarray): Input data to perform inference on.

    Returns:
        List: List containing tensors of predictions.
    """
    device = get_working_device()
    inputs = torch.from_numpy(inputs).to(device=device, dtype=torch.float)

    # Run Pytorch inference on the batch
    outputs = model(inputs)

    # Detach outputs and move to cpu
    outputs = outputs.cpu().detach()
    return outputs

class PostProcessWrapper(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 score_threshold: float = 0.001,
                 iou_threshold: float = 0.7,
                 max_detections: int = 300):
        """
        Wrapping PyTorch Module with multiclass_nms layer from sony_custom_layers.

        Args:
            model (nn.Module): Model instance.
            score_threshold (float): Score threshold for non-maximum suppression.
            iou_threshold (float): Intersection over union threshold for non-maximum suppression.
            max_detections (float): The number of detections to return.
        """
        super(PostProcessWrapper, self).__init__()
        self.model = model
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def forward(self, images):
        # model inference
        outputs = self.model(images)

        boxes = outputs[0]
        scores = outputs[1]
        nms = multiclass_nms(boxes=boxes, scores=scores, score_threshold=self.score_threshold,
                             iou_threshold=self.iou_threshold, max_detections=self.max_detections)
        return nms

def keypoints_model_predict(model: Any, inputs: np.ndarray) -> List:
    """
    Perform inference using the provided PyTorch model on the given inputs.

    This function handles moving the inputs to the appropriate torch device and data type,
    and detaches and moves the outputs to the CPU.

    Args:
        model (Any): The PyTorch model used for inference.
        inputs (np.ndarray): Input data to perform inference on.

    Returns:
        List: List containing tensors of predictions.
    """
    device = get_working_device()
    inputs = torch.from_numpy(inputs).to(device=device, dtype=torch.float)

    # Run Pytorch inference on the batch
    outputs = model(inputs)

    # Detach outputs and move to cpu
    output_np = [o.detach().cpu().numpy() for o in outputs]

    return postprocess_yolov8_keypoints(output_np)

def seg_model_predict(model: Any,
                  inputs: np.ndarray) -> List:
    """
    Perform inference using the provided PyTorch model on the given inputs.

    This function handles moving the inputs to the appropriate torch data type and format,
    and returns the outputs.

    Args:
        model (Any): The PyTorch model used for inference.
        inputs (np.ndarray): Input data to perform inference on.

    Returns:
        List: List containing tensors of predictions.
    """
    input_tensor = torch.from_numpy(inputs).unsqueeze(0)  # Add batch dimension
    device = get_working_device()
    input_tensor = input_tensor.to(device)
    # Run the model
    with torch.no_grad():
        outputs = model(input_tensor)
    outputs = [output.cpu() for output in outputs]
    return outputs

def yolov8_pytorch(model_yaml: str) -> (nn.Module, Dict):
    """
    Create PyTorch model of YOLOv8 detection.

    Args:
        model_yaml (str): Name of the YOLOv8 model configuration file (YAML format).

    Returns:
        model: YOLOv8 detection model.
        cfg_dict: YOLOv8 detection model configuration dictionary.
    """
    cfg = model_yaml
    cfg_dict = yaml_load(cfg, append_filename=True)  # model dict
    model = ModelPyTorch(cfg_dict)  # model
    return model, cfg_dict


def yolov8_pytorch_pp(model_yaml: str,
                      score_threshold: float = 0.001,
                      iou_threshold: float = 0.7,
                      max_detections: int = 300) -> (nn.Module, Dict):
    """
    Create PyTorch model of YOLOv8 detection with PostProcess.

    Args:
        model_yaml (str): Name of the YOLOv8 model configuration file (YAML format).
        score_threshold (float): Score threshold for non-maximum suppression.
        iou_threshold (float): Intersection over union threshold for non-maximum suppression.
        max_detections (float): The number of detections to return.

    Returns:
        model: YOLOv8_pp detection model.
        cfg_dict: YOLOv8_pp detection model configuration dictionary.
    """
    cfg = model_yaml
    cfg_dict = yaml_load(cfg, append_filename=True)  # model dict
    model = ModelPyTorch(cfg_dict)  # model
    model_pp = PostProcessWrapper(model=model,
                                  score_threshold=score_threshold,
                                  iou_threshold=iou_threshold,
                                  max_detections=max_detections)
    return model_pp, cfg_dict

class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        y_bb, y_cls = self.detect(self, x)

        return y_bb, y_cls, mc, p


class ModelPyTorch(nn.Module, PyTorchModelHubMixin):
    """
    Unified YOLOv8 model for both detection and segmentation.

    Args:
        cfg (dict): Model configuration in the form of a YAML string or a dictionary.
        ch (int): Number of input channels.
        mode (str): Mode of operation ('detection' or 'segmentation').
    """
    def __init__(self, cfg: dict, ch: int = 3, mode: str = 'detection'):
        super().__init__()
        self.yaml = cfg
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.mode = mode
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if isinstance(m, Segment) and self.mode == 'segmentation':
            m.inplace = self.inplace
            m.bias_init()
        elif isinstance(m, Detect) and self.mode == 'detection':
            m.inplace = self.inplace
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def make_tensors_contiguous(self):
        for name, param in self.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        for name, buffer in self.named_buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.data.contiguous()

    def save_pretrained(self, save_directory, **kwargs):
        # Make tensors contiguous
        self.make_tensors_contiguous()
        # Call the original save_pretrained method
        super().save_pretrained(save_directory, **kwargs)