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


class TreeDataset(Dataset):

    def __init__(self, csv_file, root_dir, transforms=None, label_dict={"Tree": 0}, train=True):
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
        with rasterio.open(img_name) as img:
            image = img.read()
        max_const = 40
        image[np.where(image > 40)] = max_const
        image = image / max_const
        image = np.concatenate([image, image, image])
        image = reshape_as_image(image)

        if self.train:

            # select annotations
            image_annotations = self.annotations[self.annotations.image_path ==
                                                 self.image_names[idx]]
            targets = {}
            targets["boxes"] = image_annotations[["xmin", "ymin", "xmax",
                                                  "ymax"]].values.astype(float)

            # Labels need to be encoded
            targets["labels"] = image_annotations.label.apply(
                lambda x: self.label_dict[x]).values.astype(np.int64)

            # If image has no annotations, don't augment
            if np.sum(targets["boxes"]) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.from_numpy(targets["labels"])
                # channels last
                image = np.rollaxis(image, 2, 0)
                image = torch.from_numpy(image)
                targets = {"boxes": boxes, "labels": labels}
                return self.image_names[idx], image, targets

            augmented = self.transform(image=image, bboxes=targets["boxes"], category_ids=targets["labels"])
            if len(np.array(augmented["bboxes"])) == 0:
                augmented = no_transform(image=image, bboxes=targets["boxes"], category_ids=targets["labels"])

            image = augmented["image"].type(torch.float32)

            boxes = np.array(augmented["bboxes"])
            boxes = torch.from_numpy(boxes)
            labels = np.array(augmented["category_ids"])
            labels = torch.from_numpy(labels)
            targets = {"boxes": boxes.float(), "labels": labels}

            return image, targets

        else:
            augmented = self.transform(image=image)

            return augmented["image"]
