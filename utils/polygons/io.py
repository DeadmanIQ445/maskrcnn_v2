import json
from pathlib import Path

import cv2
import fiona
import numpy as np
from shapely.geometry import Polygon, shape


def extract_polygons_from_coco(annotations_file: Path) -> dict:
    """ Parses annotation file into human-friendly format
    {
        "image_id": {
            'file_name': '1.tif',
            'polygons': [Polygon(), ...]
        }
    }
    """
    assert annotations_file.is_file()

    # Dict with image_id and polygons
    _dict = {}

    with open(annotations_file) as file:
        data = json.load(file)

    # Filling dict with image_id
    for image in data['images']:
        image_id = image['id']

        _dict[image_id] = {
            'file_name': image['file_name'],
            'polygons': []
        }

    # Filling with polygons
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        for segment in annotation['segmentation']:
            segment = np.array(segment).reshape(-1, 2)
            _dict[image_id]['polygons'].append(Polygon(segment.tolist()))

    return _dict


def masks_to_polygons(pred_masks: np.array) -> list[Polygon]:
    assert len(pred_masks.shape) == 3, "Dims of pred_masks should be [n * im_size * im_size]"

    polygons = []

    for ii in range(len(pred_masks)):
        mask = np.expand_dims(pred_masks[ii], axis=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygon = max(contours, key=cv2.contourArea).reshape(-1, 2)

        polygons.append(Polygon(polygon))

    return polygons


def read_geopolygons_from_file(path: Path):
    assert path.is_file() or path.is_dir()

    with fiona.open(path) as file:
        polygons = [shape(el['geometry']) for el in file]

    return polygons
