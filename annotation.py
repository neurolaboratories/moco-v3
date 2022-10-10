import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytz
import torch
from dateutil.parser import parse as dt_parse
from imantics import Polygons
from torch import Tensor
from PIL import Image, ImageOps


def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = ImageOps.exif_transpose(img)

    return img


@dataclass
class BBox:
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    area: float = field(init=False)

    def __post_init__(self):
        assert self.xmin >= 0 and self.xmax >= 0 and self.ymin >= 0 and self.ymax >= 0
        assert self.xmax >= self.xmin
        assert self.ymax >= self.ymin

        # the reason we add + 1 is because . assuming you have a bbox(x1,x2,y1,y2) = (1, 3, 2, 4)
        # the width of the bbox won’t be x2-x1=2, it will be how many pixels are from 1 to 3:
        #  pixel 1, pixel 2, pixel 3, or x2-x1+1
        self.area = (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)

    def iou(self, bbox) -> float:
        """iou computes intersection over union with another bbox
        Args:
            bbox (BBox): the other bbox
        Returns:
            float: intersection over union
        """
        dx = min(self.xmax, bbox.xmax) - max(self.xmin, bbox.xmin) + 1
        dy = min(self.ymax, bbox.ymax) - max(self.ymin, bbox.ymin) + 1
        if dx > 0 and dy > 0:
            intersection_area = dx * dy
            union_area = self.area + bbox.area - intersection_area
            return float(intersection_area) / union_area
        else:
            return 0.0


@dataclass
class ImageInfo:
    name: str
    path: Path = field(default=None)
    width: int = field(default=1)
    height: int = field(default=1)
    time_stamp: datetime = field(default=None)


@dataclass
class Mask:
    segmentation: List[List[int]] = field(default=None)
    width: int = field(default=1920)
    height: int = field(default=1080)

    def iou(self, other_polygons):
        mask1 = Polygons(self.segmentation).mask(self.width, self.height)
        mask2 = Polygons(other_polygons).mask(self.width, self.height)
        return mask1.iou(mask2)


@dataclass
class Annotation:
    class_id: str
    bbox: BBox
    mask: Mask
    image_info: ImageInfo
    class_name: str = field(default="")
    score: float = field(default=None)
    matched: bool = field(
        default=False)  # whether it was matched with a detection


@dataclass
class AnnotationCocoPlus:
    class_ids: List[str]
    bbox: BBox
    mask: Mask
    image_info: ImageInfo
    class_names: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    matched: bool = field(
        default=False)  # whether it was matched with a detection


def parse_coco_data(coco_file: Path) -> List[Annotation]:
    """
    Parses a coco file and returns a list of annotations.
    
    Args:
        coco_file (Path): path to the coco file.
    
    Returns:
        List[Annotation]: a list of annotations.
    """
    ann_list = []
    coco_json = json.load(coco_file.open())
    images = {c["id"]: c for c in coco_json["images"]}
    categories = {c["id"]: c for c in coco_json["categories"]}

    for ann in coco_json["annotations"]:
        xmin = int(ann["bbox"][0])
        ymin = int(ann["bbox"][1])
        xmax = xmin + int(ann["bbox"][2])
        ymax = ymin + int(ann["bbox"][3])
        img_name = images[ann["image_id"]]["file_name"]
        img_width = images[ann["image_id"]]["width"]
        img_height = images[ann["image_id"]]["height"]
        class_id = categories[ann["category_id"]]["name"]
        class_name = categories[ann["category_id"]]["supercategory"]
        segmentation = []
        score = None
        if "segmentation" in ann.keys():
            segmentation = ann["segmentation"]
        if "score" in ann.keys():
            score = ann["score"]

        timestamp = None
        try:
            timestamp = dt_parse(images[ann["image_id"]]["date_captured"])
        except Exception:
            # use None
            pass

        bbox = BBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

        mask = None
        if len(segmentation) != 0:
            mask = Mask(segmentation, img_width, img_height)

        img_info = ImageInfo(path=None,
                             name=img_name,
                             width=img_width,
                             height=img_height,
                             time_stamp=timestamp)
        ann = Annotation(class_id=class_id,
                         class_name=class_name,
                         bbox=bbox,
                         mask=mask,
                         image_info=img_info,
                         score=score)
        ann_list.append(ann)

    return ann_list


def convert_ann_to_coco(annotations: List[Annotation]) -> None:

    coco_writer = create_default_coco_writer()
    active_images = {}

    for ann in annotations:

        if ann.image_info.name in active_images:
            image_id = active_images[ann.image_info.name]
        else:
            image_info = {
                "width": ann.image_info.width,
                "height": ann.image_info.height,
                "file_name": ann.image_info.name
            }
            image_id = coco_writer.add_image(image_info)
            active_images[ann.image_info.name] = image_id

        coco_writer.add_annotations(image_id=image_id, annotations=[ann])
    return coco_writer.to_dict()


def get_image_info_from_path(
        images_path: List[Path]) -> Dict[str, Dict[str, int]]:

    file_name_to_img_info_dict = {}
    for image_path in images_path:
        im = load_image(image_path)
        width, height = im.size
        file_name = str(image_path).split("/")[-1]

        file_name_to_img_info_dict[file_name] = {
            "width": width,
            "height": height
        }

    return file_name_to_img_info_dict


def convert_model_prediction_to_coco_plus(images_results: List[List[Dict]],
                                          file_name_to_img_info_dict: Dict[
                                              str, Dict[str, int]],
                                          classes: List[str],
                                          create_all_categories=False) -> Dict:
    """Converts prediction in internal model format to COCO plus. COCO plus is the format defined
    for the metric learning approach that predicts multiple classes per annotation (top_k)

    Args:
        images_results (List[List[Dict]]): List of predictions per image 
            in internal model format {'boxes', 'scores', 'labels'}
        file_name_to_img_info_dict (Dict[ str, Dict[str, int]]): Mapping of file name to dictionary of image 
            properties {'width', 'weight'}
        classes (List[str]): List of class names
        create_all_categories (bool, optional): Flag that when set adds all categories to the
            COCO json and when unset adds only classes that have annotation. Defaults to False.

    Returns:
        Dict: Dictionary of annotation in COCO format
    """
    coco_writer = create_default_coco_writer()

    if create_all_categories:
        coco_writer.create_all_categories(classes, len(classes) * [''])

    for file_name, img_detections in zip(file_name_to_img_info_dict,
                                         images_results):

        width = file_name_to_img_info_dict[file_name]["width"]
        height = file_name_to_img_info_dict[file_name]["height"]

        image_info = {"width": width, "height": height, "file_name": file_name}
        image_id = coco_writer.add_image((image_info))

        annotations = []

        # we don't have annotations for this image
        if len(img_detections) == 0:
            continue

        for detection in img_detections:
            classes_ids = detection["labels"]
            if isinstance(classes_ids, torch.Tensor):
                classes_ids = classes_ids.tolist()
            bbox = detection["boxes"]
            scores = [float(score) for score in detection["scores"]]

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            class_indices = [
                classes[int(class_id) - 1] for class_id in classes_ids
            ]

            annotations.append(
                AnnotationCocoPlus(class_ids=class_indices,
                                   bbox=BBox(xmin=xmin,
                                             xmax=xmax,
                                             ymin=ymin,
                                             ymax=ymax),
                                   mask=None,
                                   image_info=ImageInfo(name=file_name,
                                                        path=Path(file_name),
                                                        width=width,
                                                        height=height),
                                   scores=scores))

        coco_writer.add_coco_plus_annotations(image_id=image_id,
                                              annotations=annotations)

    return coco_writer.to_dict()


def convert_model_prediction_to_coco(images_results: List[List[Dict]],
                                     file_name_to_img_info_dict: Dict[
                                         str, Dict[str, int]],
                                     classes: List[str],
                                     create_all_categories=False) -> Dict:
    """Converts internal model predictions to COCO format

    Args:
        images_results (List[List[Dict]]): List of predictions in internal 
            model format [{'labels', 'boxes', 'scores'}]
        file_name_to_img_info_dict (Dict[ str, Dict[str, int]]): Dictionary that maps 
            image name to image properties: width and height
        classes (List[str]): List of ordered class names
        create_all_categories (bool, optional): Flag that when set adds all categories to the
            COCO json and when unset adds only classes that have annotation. Defaults to False.

    Returns:
        Dict: Predictions in COCO format
    """
    coco_writer = create_default_coco_writer()

    if create_all_categories:
        coco_writer.create_all_categories(classes, len(classes) * [''])

    for file_name, img_detections in zip(file_name_to_img_info_dict,
                                         images_results):

        width = file_name_to_img_info_dict[file_name]["width"]
        height = file_name_to_img_info_dict[file_name]["height"]

        image_info = {"width": width, "height": height, "file_name": file_name}
        image_id = coco_writer.add_image((image_info))

        annotations = []

        # we don't have annotations for this image
        if len(img_detections) == 0:
            continue

        for detection in img_detections:
            class_id = detection["labels"]
            if isinstance(class_id, torch.Tensor):
                class_id = class_id.item()
            bbox = detection["boxes"]
            score = detection["scores"]

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            score = float(score)
            annotations.append(
                Annotation(class_id=classes[int(class_id) - 1],
                           bbox=BBox(xmin=xmin,
                                     xmax=xmax,
                                     ymin=ymin,
                                     ymax=ymax),
                           mask=None,
                           image_info=ImageInfo(name=file_name,
                                                path=Path(file_name),
                                                width=width,
                                                height=height),
                           score=score))

        coco_writer.add_annotations(image_id=image_id, annotations=annotations)

    return coco_writer.to_dict()


def _extract_bounding_box(annotation: Annotation) -> List:
    xmin = annotation.bbox.xmin
    xmax = annotation.bbox.xmax
    ymin = annotation.bbox.ymin
    ymax = annotation.bbox.ymax
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def _extract_area(annotation: Annotation) -> int:
    width = annotation.bbox.xmax - annotation.bbox.xmin
    height = annotation.bbox.ymax - annotation.bbox.ymin
    return width * height


class CocoWriter:
    def __init__(self, info):
        self.info = info
        self.licenses = [self._default_license()]
        self.images = []
        self.annotations = []
        self.categories = []

        self.curr_annotation_idx = 0
        self.curr_image_idx = 0
        self.curr_category_idx = 0
        self.reverse_category_idx = {}

    def _default_license(self):
        return {
            "url": "",
            "id": 1,
            "name": "",
        }

    def _get_license(self):
        return 1

    def add_image(self, image_info: Dict) -> int:
        """ adds the image info to the CocoWriter to track them
        """
        image_id = self._next_image_id()

        # extend image_info
        image_info.update({
            "coco_url": "",
            "flickr_url": "",
            "license": str(1),
            "id": image_id,
            "date_captured": ""
        })
        self.images.append(image_info)

        return image_id

    def add_annotations(self, image_id: int,
                        annotations: List[Annotation]) -> None:
        """ adds to the CocoWriter the annotations tied to an image
        """
        for annotation in annotations:
            category_name = annotation.class_id
            self._create_if_missing(category_name=category_name,
                                    supercategory=annotation.class_name)

            self.annotations.append({
                "id":
                self._next_annotation_id(),
                "image_id":
                image_id,
                "segmentation":
                annotation.mask.segmentation if annotation.mask else [],
                "category_id":
                self.reverse_category_idx[category_name],
                "bbox":
                _extract_bounding_box(annotation),
                "area":
                _extract_area(annotation),
                "iscrowd":
                0,
                "score":
                annotation.score
            })

    def add_coco_plus_annotations(
            self, image_id: int,
            annotations: List[AnnotationCocoPlus]) -> None:
        """ adds to the CocoWriter the annotations tied to an image
        """
        for annotation in annotations:
            category_names = annotation.class_ids

            for category_name in category_names:
                self._create_if_missing(category_name=category_name,
                                        supercategory=annotation.class_names)

            reverse_category_ids = [
                self.reverse_category_idx[category_name]
                for category_name in category_names
            ]

            self.annotations.append({
                "id":
                self._next_annotation_id(),
                "image_id":
                image_id,
                "segmentation":
                annotation.mask.segmentation if annotation.mask else [],
                "category_id":
                reverse_category_ids,
                "bbox":
                _extract_bounding_box(annotation),
                "area":
                _extract_area(annotation),
                "iscrowd":
                0,
                "score":
                annotation.scores
            })

    def create_all_categories(self, category_names, supercategories):
        for category_name, supercategory in zip(category_names,
                                                supercategories):
            self._create_if_missing(category_name, supercategory)

    def _create_if_missing(self, category_name, supercategory):
        if category_name not in self.reverse_category_idx.keys():
            category = {
                "id": self._next_category_id(),
                "name": category_name,  # hack -> uuid
                "supercategory": supercategory,
            }
            self.categories.append(category)
            self.reverse_category_idx[category_name] = category['id']

    def _next_annotation_id(self) -> int:
        self.curr_annotation_idx += 1
        return self.curr_annotation_idx

    def _next_image_id(self) -> int:
        self.curr_image_idx += 1
        return self.curr_image_idx

    def _next_category_id(self) -> int:
        self.curr_category_idx += 1
        return self.curr_category_idx

    def _sort_alphabetically(self) -> None:
        sorted_c = sorted(self.categories, key=lambda x: x["name"])
        old2new = {}
        new_index = 0
        for c in sorted_c:
            old_index = c["id"]
            new_index += 1
            old2new[old_index] = new_index
        for c in sorted_c:
            c["id"] = old2new[c["id"]]
        for ann in self.annotations:
            if isinstance(ann["category_id"], List):
                ann["category_id"] = [
                    old2new[ann_category_id]
                    for ann_category_id in ann["category_id"]
                ]
            else:
                ann["category_id"] = old2new[ann["category_id"]]
        self.categories = sorted_c

    def to_json_string(self):
        self._sort_alphabetically()
        return json.dumps(self.to_dict())

    def to_dict(self):
        self._sort_alphabetically()
        return {
            "info": self.info,
            "licenses": self.licenses,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }


def create_default_coco_writer() -> CocoWriter:
    tz = pytz.timezone('Europe/London')
    now = datetime.now(tz)
    timenow = now.strftime("%d/%m/%Y %H:%M:%S")

    coco_writer = CocoWriter({
        "year": "2021",
        "version": "1",
        "description": "",
        "contributor": "",
        "url": "",
        "date_created": timenow
    })
    return coco_writer


def pretty_dump_json(coco_dict: Dict, coco_file_path: Path) -> None:
    with open(str(coco_file_path), "w") as f:
        json.dump(coco_dict, f, indent=4, sort_keys=True)


def convert_tensor_to_ann_dict(
        input_tensor: List[Tensor]) -> List[Dict[str, Tensor]]:

    predictions = []
    for pred_per_image in input_tensor:
        per_image_preds = {}

        per_image_preds['boxes'] = pred_per_image[:4]
        per_image_preds['scores'] = pred_per_image[4]
        per_image_preds['labels'] = pred_per_image[5]

        predictions.append(per_image_preds)

    return predictions