from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as albu
import pandas as pd
import cv2
import rasterio
from rasterio.plot import reshape_as_image


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
        return albu.Compose([ToTensorV2()],
            bbox_params=albu.BboxParams(format='pascal_voc', label_fields=["category_ids"]))


no_transform = get_training_augmentation(augment=False)


import os
import numpy as np
import torch


class PennFudanDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms):
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
        img = reshape_as_image(image)
        with rasterio.open(mask_path) as f:
            mask = reshape_as_image(f.read())
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = np.array([mask==id for id in obj_ids])
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
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        if boxes.shape[0]==0:
            target = {}
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(1, dtype=torch.int64)
            target["image_id"] = image_id
            target["area"] = torch.zeros(1, dtype=torch.float32)
            target["iscrowd"] = torch.zeros(1, dtype=torch.float32)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        non_zero = torch.where(area>1)
        target = {}
        target["boxes"] = boxes[non_zero]
        target["labels"] = labels[non_zero]
        target["image_id"] = image_id
        target["area"] = area[non_zero]
        target["iscrowd"] = iscrowd[non_zero]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_training_augmentation(augment=True):
    if augment:
        scale_limit = (-0.2, 0.5)
        return albu.Compose([
            # albu.ShiftScaleRotate(shift_limit=0, rotate_limit=180,
            #                       scale_limit=scale_limit,
            #                       interpolation=cv2.INTER_NEAREST,
            #                       ),
            # #                     albu.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05),
            # #                     albu.MultiplicativeNoise(multiplier=(0.92, 1.1)),
            #
            # albu.Flip(),
            # albu.PiecewiseAffine(scale=(0.01, 0.05), p=0.2),
            ToTensorV2()],
            bbox_params=albu.BboxParams(format='pascal_voc', label_fields=["category_ids"]))
    else:
        return albu.Compose([ToTensorV2()],albu.BboxParams(format='pascal_voc', label_fields=["category_ids"]))


no_transform = get_training_augmentation(augment=False)

perc = 0.1
class TreeDataset(Dataset):

    def __init__(self, csv_file, root_dir, transforms=None, label_dict={"Tree": 1}, train=True):
        """
        Args:
            csv_file (string): Path to a single csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            label_dict: a dictionary where keys are labels from the csv column and values are numeric labels "Tree" -> 0
        Returns:
            If train:
                path, image, targets
            else:
                image
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transforms is None:
            self.transform = get_training_augmentation(augment=train)
        else:
            self.transform = transforms
        self.image_names = self.annotations.image_path.unique()
        self.label_dict = label_dict
        self.train = train

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        # read, scale and set to float
        with rasterio.open(img_name) as f:
            image = f.read()

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
        # image = np.stack([image[0],image[0],image[0]])
        image = reshape_as_image(image)
        image_annotations = self.annotations[self.annotations.image_path ==
                                             self.image_names[idx]]
        targets = {}
        targets["boxes"] = image_annotations[["xmin", "ymin", "xmax",
                                              "ymax"]].values.astype(float)

        targets["labels"] = image_annotations.label.apply(
            lambda x: self.label_dict[x]).values.astype(np.int64)
        image_id = torch.tensor([idx])

        if self.train:
            # If image has no annotations, don't augment
            if np.sum(targets["boxes"]) == 0:
                target = {}
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros(1, dtype=torch.int64)
                target["image_id"] = image_id
                target["area"] = torch.zeros(1, dtype=torch.float32)
                target["iscrowd"] = torch.zeros(1, dtype=torch.float32)
                return torch.tensor(image), target

            augmented = self.transform(image=image, bboxes=targets["boxes"], category_ids=targets["labels"])
            if len(np.array(augmented["bboxes"])) == 0:
                target = {}
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros(1, dtype=torch.int64)
                target["image_id"] = image_id
                target["area"] = torch.zeros(1, dtype=torch.float32)
                target["iscrowd"] = torch.zeros(1, dtype=torch.float32)
                return augmented["image"].type(torch.float32), target, img_name

        else:
            augmented = self.transform(image=image, bboxes=targets["boxes"], category_ids=targets["labels"])

        image = augmented["image"].type(torch.float32)

        boxes = np.array(augmented["bboxes"])
        boxes = torch.from_numpy(boxes)
        labels = np.array(augmented["category_ids"])
        labels = torch.from_numpy(labels)
        # boxes_x = boxes[:, 3] - boxes[:, 1]
        # boxes[:, 3] = boxes[:, 3] - boxes_x*perc
        # boxes[:, 1] = boxes[:, 1] + boxes_x*perc
        # boxes_y = boxes[:, 2] - boxes[:, 0]
        # boxes[:, 2] = boxes[:, 2] - boxes_y*perc
        # boxes[:, 0] = boxes[:, 0] + boxes_y*perc
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        return image, target, img_name
