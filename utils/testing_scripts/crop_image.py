import datetime
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from utils.preprocess import pycococreatortools
from utils.preprocess.tif_process import GeoTiff

INFO = {
    "description": "ZemleUchastki",
    "url": "",
    "version": "0.1.1",
    "year": 2021,
    "contributor": "Ibragim, Shamil",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]
CATEGORIES = [
    {
        'id': 1,
        'name': 'uchastok',
        'supercategory': 'land',
    },
]


def from_mask_to_coco(image_path, annotation_path, json_output):
    image_filename = image_path

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    image = Image.open(image_filename)

    image_info = pycococreatortools.create_image_info(
        image_id, os.path.basename(image_filename), image.size
    )

    coco_output['images'].append(image_info)

    for annotation_filename in Path(annotation_path).glob("**/*"):
        annotation_filename = str(annotation_filename)
        print(annotation_filename)
        if '.ipynb_checkpoints' in str(annotation_filename):
            continue

        class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
        binary_mask = np.asarray(Image.open(annotation_filename)
                                 .convert('1')).astype(np.uint8)

        annotation_info = pycococreatortools.create_annotation_info(
            segmentation_id, image_id, category_info, binary_mask,
            image.size, tolerance=2)

        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)

        segmentation_id = segmentation_id + 1

    with open(json_output, 'w') as output_json:
        json.dump(coco_output, output_json)


def clip_from_file(clip_size, root, image_name, img_path="Raster", shp_path="Razmetka", ):
    pic_id = 0

    tiff = GeoTiff(os.path.join(root, img_path, image_name))
    img_id = image_name.split(".", 1)[0]
    print(f"img_id: {img_id}")
    tiff.clip_tif_and_shapefile(
        clip_size=clip_size,
        begin_id=pic_id,
        shapefile_path=os.path.join(root, shp_path, img_id, img_id + '.shp'),
        out_dir=os.path.join(root, f'size{clip_size}')
    )
