from __future__ import division

import argparse
import copy
import importlib
import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.experimental import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image, ImageColor, ImageDraw, ImageFont
from pycocotools import coco
from torch import Tensor
from tqdm import tqdm

from annotation import (convert_model_prediction_to_coco,
                        get_image_info_from_path, pretty_dump_json, load_image)


def convert_bbox_from_xyxy_to_xywh(bbox: Tensor) -> Tensor:
    return torch.tensor(
        [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])


def create_cropped_images(
        results: List[Dict],
        pil_img: Image,
        boxes_format: str = 'xyxy',
        labels: bool = False,
        enlarge_fraction: float = 0) -> Tuple[List[Any], Path]:
    """create_cropped_images Generates cropped images according to predictions ofa generic object detector. 
    Args:
        results (List[Dict]): List of dictionaries containing detections/image.
        pil_img (Image): Image loaded in PIL format.
        boxes_format (str): Current format of the boxes, if xywh, then it will be changed to xyxy.
        labels (bool): Whether to return labels too.
        enlarge_fraction (float): how much to enlarge each box, by percentage of w or h.

    Returns:
        Dict[List[Image]], Path  : Tuple of list of newly created cropped images and directory where they are saved
    """
    cropped_imgs = []
    if labels:
        cropped_labels = []
    for detection in results:
        if detection['boxes'][2] < 1 or detection['boxes'][3] < 1:
            continue
        crop_detections = detection['boxes'][:]
        if boxes_format == 'xywh':
            crop_detections = xywh_to_xyxy(crop_detections)
        if enlarge_fraction > 0:
            crop_detections = enlarge_bbox(crop_detections, enlarge_fraction)
        cropped_img = crop_image(pil_img, crop_detections)
        cropped_imgs.append(cropped_img)
        if labels:
            cropped_labels.append(detection['labels'])
    if labels:
        return cropped_imgs, cropped_labels
    else:
        return cropped_imgs


def create_crops_from_list_of_annotations(
        annotation_list: list,
        image_id_to_pil_dict: dict,
        img_id_list: list,
        boxes_format: str = 'xyxy',
        enlarge_fraction: float = 0) -> List[Any]:
    """create_cropped_images Generates cropped images according to predictions ofa generic object detector. 
    Args:
        annotation_list (list): list of annotations, in same order as in input coco file.
        image_id_to_pil_dict (dict): dict of image ids and pil imgs.
        img_id_list (list): list of image id's for each annotation.
        boxes_format (str): Current format of the boxes, if xywh, then it will be changed to xyxy.
        enlarge_fraction (float): how much to enlarge each box, by percentage of w or h.

    Returns:
        List[Image]: list of newly created crops per image and per annotation 
    """
    cropped_imgs = []
    for detection, img_id in zip(annotation_list, img_id_list):
        if detection['boxes'][2] < 1 or detection['boxes'][3] < 1:
            continue
        crop_detections = detection['boxes'][:]
        if boxes_format == 'xywh':
            crop_detections = xywh_to_xyxy(crop_detections)
        if enlarge_fraction > 0:
            crop_detections = enlarge_bbox(crop_detections, enlarge_fraction)
        cropped_img = crop_image(image_id_to_pil_dict[img_id], crop_detections)
        cropped_imgs.append(cropped_img)

    return cropped_imgs


def enlarge_bbox(boxes: List[float], fraction: float) -> List[float]:
    """enlarge_bbox takes a bounding box in format [x_1, y_1, x_2, y_2],
     takes fraction*width/height of the bounding box
     and makes the box that much larger.
    Args:
        boxes (List[float]): Box in format [x_1, y_1, x_2, y_2].
        fraction (float): How much of width/height to augment.
    Returns:
        List[float]: Enlarged box in format [x_1, y_1, x_2, y_2]
    """
    w = boxes[2] - boxes[0]
    h = boxes[3] - boxes[1]
    w_2 = int(fraction * w)
    h_2 = int(fraction * h)
    boxes[0] = boxes[0] + w_2
    boxes[1] = boxes[1] + h_2
    boxes[2] = boxes[2] + w_2
    boxes[3] = boxes[3] + h_2
    return boxes


def xywh_to_xyxy(boxes: List) -> List:
    x_1 = boxes[0]
    y_1 = boxes[1]
    x_2 = boxes[0] + boxes[2]
    y_2 = boxes[1] + boxes[3]
    return [x_1, y_1, x_2, y_2]


def crop_image(img: Image, bbox: torch.Tensor) -> Image:
    """crop_image Crops an image using a bbox
    Args:
        img (Image): full image to crop 
        bbox (torch.Tensor): bbox to use to crop image, tensor of size 4
    Returns:
        Image: cropped PIL Image
    """
    return img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))


def PILuint8_to_float32(pil_img: Image) -> np.float32:
    """PILuint8_to_float32 Transforms PIL image in range [0,255] in float32 array in range [0,1]
    Args:
        pil_img (PIL.Image): a PIL Image
    Returns:
        [np.float32]: Array representation of PIL Image
    """
    return (np.asarray(pil_img).astype(np.float32)) / (255.)


def resize_tensor_box(bbox: Tensor, in_size: Tuple, out_size: Tuple) -> Tensor:
    """Resize bounding boxes according to image resize operation. Tensor operations
    Parameters
    ----------
    bbox : torch.FloatTensor
        Torch.FloatTensor with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    in_size : tuple
        Tuple of length 2: (width, height) for input.
    out_size : tuple
        Tuple of length 2: (width, height) for output.
    Returns
    -------
    torch.FloatTensor
        Resized bounding boxes with original shape.
    """
    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(
            len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(
            len(out_size)))
    x_scale = out_size[0] / in_size[0]
    y_scale = out_size[1] / in_size[1]
    # Below is equivalent to what happens in resize box numpy.
    return bbox * torch.FloatTensor([x_scale, y_scale, x_scale, y_scale]).to(
        bbox.device)


def resize_box(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize operation.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    in_size : tuple
        Tuple of length 2: (width, height) for input.
    out_size : tuple
        Tuple of length 2: (width, height) for output.
    Returns
    -------
    numpy.ndarray
        Resized bounding boxes with original shape.
    """
    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(
            len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(
            len(out_size)))
    bbox = bbox.copy().astype(float)
    x_scale = out_size[0] / in_size[0]
    y_scale = out_size[1] / in_size[1]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 4] = y_scale * bbox[:, 4]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def imshow_tensor(img: torch.Tensor) -> plt:
    """imshow_tensor Takes an image with 3 channels and displays it correctly. 
    Args:
        img (torch.Tensor): An image represented as Tensor with dimensions [C, W, H]
    Returns:
        matplolib.plot: Displays image using Matplotlib 
    """
    return plt.imshow(img.permute(1, 2, 0))


def draw_img_annotations(img_path: Path,
                         anns: List[Dict],
                         colour: Dict,
                         classes: List[str],
                         out: Optional[Path] = None,
                         plot_score: bool = True) -> Image:
    """draw_img_annotations Draws annotations on image Tensor using Tensor boxes as input
    Args:
        img (torch.Tensor): A Tensor of type [C, W, H]
        anns (Dict[torch.Tensor, torch.Tensor]): A Dict with box Tensors [N, 4] and labels Tensors [N]
        colour (str): The border colour of the bboxes
        out (Optional[Path], optional): The file to save the image. Defaults to None.
    Returns:
        [Image]: Pil Image with drawn bboxes and label number
    """
    pil_img = load_image(img_path)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("fonts/arial.ttf", 25)
    for ann in anns:
        bbox = ann['boxes']
        label = ann['labels']
        score = ann['scores']
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        if isinstance(score, torch.Tensor):
            score = score.cpu().numpy()
        score = np.around(score * 100, 1)

        draw.rectangle(list(bbox),
                       outline=colour[classes[int(label) - 1]],
                       fill=None,
                       width=5)
        if plot_score:
            text = classes[int(label) - 1] + " " + str(score) + "%"
        else:
            text = classes[int(label) - 1]
        draw.text((bbox[0], bbox[1]),
                  text,
                  fill=colour[classes[int(label) - 1]],
                  anchor="ls",
                  font=font)
    if out is not None:
        pil_img.save(out)
        return pil_img
    else:
        return pil_img


def plot_annotations(detections: List[Dict], images_path: str,
                     output_location: str, classes: List[str],
                     plot_score: bool) -> None:
    dir_path = Path(output_location) / "img"
    dir_path.mkdir(parents=True, exist_ok=True)
    colors = list(ImageColor.colormap.items())
    color_per_classes = {}
    for class_item in classes:
        color_per_classes.update(
            {class_item: colors[random.randint(0,
                                               len(colors) - 1)][0]})
    for i in range(0, len(images_path)):
        image_name = str(images_path[i]).split("/")[-1]
        out_path = dir_path / image_name
        draw_img_annotations(img_path=images_path[i],
                             anns=detections[i],
                             colour=color_per_classes,
                             classes=classes,
                             out=out_path,
                             plot_score=plot_score)


def convert_xywh_to_xyxy(bbox: torch.Tensor) -> torch.tensor:
    """ Converts a batch of bboxes from [xmin, ymin, width, height] 
    to [xmin, ymin, xmax, ymax]
    Args:
        bbox (torch.Tensor): torch tensor representing annotation
    Returns:
        [torch.tensor]: Converted bbox to xyxy
    """
    converted_bbox = deepcopy(bbox)
    converted_bbox[:, 2] += bbox[:, 0]
    converted_bbox[:, 3] += bbox[:, 1]
    return converted_bbox


def convert_xywh_to_yxyx(bbox: torch.Tensor) -> torch.tensor:
    """convert_xywh_to_yxyx Converts a batch of bboxes from [xmin, ymin, width, height] 
        to [ymin, xmin, ymax, xmax]
        Args:
        bbox (torch.Tensor): torch tensor representing annotation
        Returns:
        [torch.tensor]: Converted bbox to yxyx
    """
    converted_bbox = deepcopy(bbox)
    converted_bbox[:, 2] += bbox[:, 0]
    converted_bbox[:, 3] += bbox[:, 1]
    aux_bbox = deepcopy(bbox)
    aux_bbox[:, 0] = converted_bbox[:, 1]
    aux_bbox[:, 1] = converted_bbox[:, 0]
    aux_bbox[:, 2] = converted_bbox[:, 3]
    aux_bbox[:, 3] = converted_bbox[:, 2]
    return aux_bbox


def load_obj(obj_path: str, default_obj_path: str = '') -> Any:
    """
    Function taken from: https://github.com/Erlemar/wheat/blob/master/src/utils/utils.py#L15
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(
        0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)


def check_and_correct_bbox_annotations(img, bbox):
    # check if the coordinates xmax, ymax are outside of the image and correct them
    if (bbox[0] + bbox[2]) > img.size[0]:
        bbox[2] = img.size[0] - bbox[0]
    if (bbox[1] + bbox[3]) > img.size[1]:
        bbox[3] = img.size[1] - bbox[1]
    # check if the coordinates xmin, ymin are outside of the image and correct them
    if bbox[0] > img.size[0]:
        bbox[0] = img.size[0]
    if bbox[1] > img.size[1]:
        bbox[1] = img.size[1]
    return bbox


def convert_obj_det_dataset_to_classif_dataset(
        # dataset local path /valohai/.../dataset
        dataset_path: Path,
        new_dataset_path: Path,
        dataset_type: str,
        max_crops_per_class: int = 10,
        enlarge_boxes_fraction: float = 0.0,
        remove_unknown: bool = False) -> None:

    if not isinstance(dataset_path, Path):
        dataset_path = Path(dataset_path)

    if not isinstance(new_dataset_path, Path):
        new_dataset_path = Path(new_dataset_path)

    json_path = dataset_path / ("coco_" + dataset_type + ".json")
    with open(json_path, "r") as f:
        coco_data = json.load(f)
    pycoco = coco.COCO(json_path)
    images = coco_data['images']

    per_class_n = defaultdict(int)

    for img in tqdm(images, desc="converting " + dataset_type + " data"):
        opened_img = load_image(dataset_path / Path("./img/") /
                                img['file_name'])
        anns = pycoco.imgToAnns[img["id"]]
        copied_img = copy.deepcopy(opened_img)
        for ann in anns:
            # check if the bbox annotations are correct
            category_name = pycoco.cats[ann['category_id']]['name']

            if remove_unknown and (category_name == "object"
                                   or category_name == "unknown"):
                continue

            if per_class_n[category_name] >= max_crops_per_class:
                continue

            if ann['bbox'][2] == 0 or ann['bbox'][3] == 0:
                continue

            bbox = check_and_correct_bbox_annotations(copied_img, ann['bbox'])
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            if enlarge_boxes_fraction > 0:
                bbox = enlarge_bbox(bbox, enlarge_boxes_fraction)
            cropped_img = crop_image(copied_img, bbox)
            new_img_path = new_dataset_path / Path(
                "./" + dataset_type) / category_name.lower().replace(' ', '_')
            object_img_path = new_dataset_path / Path(
                "./" + dataset_type + '_object') / 'object'
            new_img_path.mkdir(exist_ok=True, parents=True)
            object_img_path.mkdir(exist_ok=True, parents=True)
            cropped_img.save(new_img_path / Path(str(ann['id']) + ".png"))
            cropped_img.save(object_img_path / Path(str(ann['id']) + ".png"))
            per_class_n[category_name] += 1


def get_classes_from_coco(dataset_path: Path,
                          dataset_type: str = None,
                          remove_unknown: bool = False) -> List[str]:

    if not isinstance(dataset_path, Path):
        dataset_path = Path(dataset_path)

    if dataset_type:
        json_path = dataset_path / ("coco_" + dataset_type + ".json")

    else:
        json_path = dataset_path

    if not json_path.is_file():
        return None

    with open(json_path, "r") as f:
        coco_data = json.load(f)

    classes = [
        category['name'] for category in coco_data['categories']
        if (not remove_unknown) or (remove_unknown and (
            category['name'] != 'object') or (category['name'] != 'unknown'))
    ]

    return sorted(classes)


def get_coco_classes() -> List[Dict]:
    coco_cfg_path = Path.cwd() / 'zia_vision' / 'configs' / 'coco_classes.json'
    with open(coco_cfg_path) as coco_cfg:
        coco_cfg = json.load(coco_cfg)
    classes = coco_cfg['classes']

    return classes


def save_predictions(final_results: List[List[Dict]], images_path: List[Path],
                     classes: List, output_location: str, plot_score: bool,
                     create_plots: bool):

    images_info = get_image_info_from_path(images_path)
    coco_final_results = convert_model_prediction_to_coco(final_results,
                                                          images_info,
                                                          classes=classes)
    pretty_dump_json(coco_final_results,
                     Path(output_location) / 'coco_test.json')

    if create_plots:
        plot_annotations(final_results,
                         images_path,
                         output_location=output_location,
                         classes=classes,
                         plot_score=plot_score)


def check_predictions_for_classes(prediction: Dict, classes: List) -> bool:
    """Checks if predictions contain any classes from a list

    Args:
        prediction (Dict): Model predictions in deployment format
        classes (List): List of classes to check for in predictions

    Returns:
        bool: Presence of filtered classes in predictions
    """
    for pred in prediction:
        if pred['labels'] in classes:
            return True
    return False


def override_base_configs(base_config_yaml: Path) -> OmegaConf:
    #1. Set-up and Initialize all configs
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    # Hydra initializes from dir and wants top level directory. Look a bit more into this.
    # Composing with argparse will take everything from cli and overwrite base yaml config
    initialize_config_dir(config_dir=str(base_config_yaml.parent))
    cfg = compose(base_config_yaml.name, overrides=args.overrides, strict=True)

    return cfg


def convert_coco_ann_to_deploy_ann(ann_coco: Dict,
                                   category_id_to_name: Dict) -> Dict:
    deploy_dict = {}

    deploy_dict['boxes'] = ann_coco['bbox']

    deploy_dict['scores'] = 1.0

    if 'score' in ann_coco:
        deploy_dict['scores'] = ann_coco['score']

    if 'neurolabs' in ann_coco and 'score' in ann_coco['neurolabs']:
        deploy_dict['scores'] = ann_coco['neurolabs']['score']

    deploy_dict['labels'] = category_id_to_name[ann_coco['category_id']]

    return deploy_dict


def convert_coco_to_internal_format(coco_dict: Dict) -> Dict[str, List[Dict]]:
    """Converts COCO annotation to internal model format

    Args:
        coco_dict (Dict): Predictions dict in COCO format

    Returns:
        Dict[str, List[Dict]]: Predictions in internal model format ['image' :[{'scores', 'labels', 'boxes'}] 
    """

    category_id_to_name = {}
    for category in coco_dict['categories']:
        category_id_to_name[category['id']] = category['name']

    image_id_to_name = {}
    for img in coco_dict['images']:
        image_id_to_name[img['id']] = img['file_name']

    deployment_response_per_image = {}

    for annotation in coco_dict['annotations']:

        deploy_dict = convert_coco_ann_to_deploy_ann(annotation,
                                                     category_id_to_name)

        file_name = image_id_to_name[annotation['image_id']]

        if file_name not in deployment_response_per_image:
            deployment_response_per_image[file_name] = []

        deployment_response_per_image[file_name].append(deploy_dict)

    return deployment_response_per_image


def convert_coco_to_internal_list_of_annot(
        coco_dict: Dict) -> Tuple[list, list]:
    """Converts COCO annotation to internal format: list of annotations and list of image ids.

    Args:
        coco_dict (Dict): Predictions dict in COCO format

    Returns:
        Tuple[list, list]: list of annotations and list of image ids. 
        Annotations in internal model format ['image' :[{'scores', 'labels', 'boxes'}] 
    """

    category_id_to_name = {}
    for category in coco_dict['categories']:
        category_id_to_name[category['id']] = category['name']

    annotation_list = []
    img_id_list = []

    for annotation in coco_dict['annotations']:

        deploy_dict = convert_coco_ann_to_deploy_ann(annotation,
                                                     category_id_to_name)

        annotation_list.append(deploy_dict)
        img_id_list.append(annotation['image_id'])

    return annotation_list, img_id_list


def convert_coco_plus_to_coco(coco_plus_annotations: List[List[Dict]]):
    coco_annotations = []

    for image_anns in coco_plus_annotations:
        new_image_anns = []
        for ann in image_anns:
            new_ann = {
                'boxes': ann['boxes'],
                'scores': ann['scores'][0],
                'labels': ann['labels'][0]
            }
            new_image_anns.append(new_ann)
        coco_annotations.append(new_image_anns)

    return coco_annotations