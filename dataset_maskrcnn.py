import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as albu
import numpy as np
import pandas as pd
import cv2
import rasterio
from rasterio.plot import reshape_as_image
import os

def get_training_augmentation(augment=True):
    if augment:
        scale_limit = (-0.2, 0.5)
        return albu.Compose([
            albu.ShiftScaleRotate(shift_limit=0, rotate_limit=180,
                                  scale_limit=scale_limit,
                                  interpolation=cv2.INTER_NEAREST,
                                  ),
            #                     albu.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05),
            #                     albu.MultiplicativeNoise(multiplier=(0.92, 1.1)),

            albu.Flip(),
            albu.PiecewiseAffine(scale=(0.01, 0.05), p=0.2),
            ToTensorV2()],
            bbox_params=albu.BboxParams(format='pascal_voc', label_fields=["category_ids"]))
    else:
        return albu.Compose([ToTensorV2()])


no_transform = get_training_augmentation(augment=False)


def preprocess_rgb(x):
    return x.astype(float)/255

def preprocess_chm_treecanopy(image):
    image[0] = image[0] / 40
    max_its = np.max(image[1])
    if max_its > 1.0:
        if max_its < 255:
            image[1] = image[1] / 255
        else:
            image[1] = image[1] / max_its
    image[2] = image[2] / 2.0
    image[image > 1] = 1
    image[image < 0] = 0
    return image


class MaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, preprocess):
        self.preprocess = preprocess
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        
        with rasterio.open(img_path) as f:
            image = f.read()
            image = self.preprocess(image).astype(np.float32)


        img = reshape_as_image(image)
        with rasterio.open(mask_path) as f:
            mask = reshape_as_image(f.read())
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        mask = mask.squeeze()
        masks = np.array([mask==id for id in obj_ids])

        # transformed = self.transforms(image=img, masks=masks)
        # img = transformed['image']
        # mask = transformed['masks']
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])




        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float16)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        if boxes.shape[0]==0:
            target = {}
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float16)
            target["labels"] = torch.zeros(1, dtype=torch.int64)
            target["masks"] = torch.tensor(np.expand_dims(mask, axis=0), dtype=torch.int64)
            target["image_id"] = image_id
            target["area"] = torch.zeros(1, dtype=torch.float16)
            target["iscrowd"] = torch.zeros(1, dtype=torch.float16)
            if self.transforms is not None:
                transform = self.transforms(image=img, target=target)
                img = transform['image']
                target = transform['target']
            return img.float(), target
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        non_zero = torch.where(area>1)
        target = {}
        target["boxes"] = boxes[non_zero]
        target["labels"] = labels[non_zero]
        target["masks"] = masks[non_zero]
        target["image_id"] = image_id
        target["area"] = area[non_zero]
        target["iscrowd"] = iscrowd[non_zero]
        if self.transforms is not None:
            transform = self.transforms(image=img, target=target)
            img = transform['image']
            target= transform['target']
        return img.float(), target

    def __len__(self):
        return len(self.imgs)