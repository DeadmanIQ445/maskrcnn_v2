import os
import time
import torch

import torchvision.models.detection.mask_rcnn

from references.coco_utils import get_coco_api_from_dataset
from references.coco_eval import CocoEvaluator
from references import utils



def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device,epoch):

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        gt = {target["image_id"].item(): target for target in targets}
        save_img(images, res, epoch)
        save_img_mask(images, res, epoch, save_path="output_mask")
        save_img(images, gt,1,"output_gt")
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

import cv2
from rasterio.plot import reshape_as_image
import numpy as np
def save_img(images,res,n_epoch, save_path='output_sample'):

    for k,v in res.items():
        img = reshape_as_image(images[0].cpu().numpy())
        img= img*255
        img = img.astype(np.uint8).copy()
        out = v['boxes'].numpy()
        for i in range(out.shape[0]):
            boundRect = out[i]
            cv2.rectangle(img, (int(boundRect[0]), int(boundRect[1])),
                         ( int(boundRect[2]), int(boundRect[3])), (255,0,0), 2)
        os.makedirs(save_path+f"/{n_epoch}", exist_ok=True)
        cv2.imwrite(save_path+f"/{n_epoch}/{k}.jpg",img)

def save_img_mask(images,res,n_epoch, save_path='output_sample'):

    for k,v in res.items():
        img = reshape_as_image(images[0].cpu().numpy())
        img= img*255
        img = img.astype(np.uint8).copy()
        out = v['masks'].numpy()
        for i in range(out.shape[0]):
            mask = out[i].squeeze()
            # print(np.unique(mask))
            img[mask>0.5,:] = i
            # res = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
            os.makedirs(save_path + f"/mask", exist_ok=True)
            # cv2.imwrite(save_path + f"/mask/{i}.jpg", reshape_as_image(out[i])*255)
        os.makedirs(save_path+f"/{n_epoch}", exist_ok=True)
        cv2.imwrite(save_path+f"/{n_epoch}/{k}.jpg",img)

import glob
import rasterio
import pandas as pd
if __name__ == '__main__':
    samples_dir = '/media/deadman445/disk/Add/summer_tiles_400'
    save_path='/media/deadman445/disk/Add/test_vis_bbox/'
    for i in glob.glob(samples_dir+"/*.tif"):
        with rasterio.open(i) as f:
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
        img= img*255
        img = img.astype(np.uint8).copy()
        annotations = pd.read_csv(samples_dir+'/final_df.csv')
        image_annotations = annotations[annotations.image_path == i.split('/')[-1]]
        targets = {}
        targets["boxes"] = image_annotations[["xmin", "ymin", "xmax",
                                              "ymax"]].values.astype(float)
        out = targets["boxes"]
        for j in range(out.shape[0]):
            boundRect = out[j]
            cv2.rectangle(img, (int(boundRect[0]), int(boundRect[1])),
                         ( int(boundRect[2]), int(boundRect[3])), (255,0,0), 2)
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(save_path+f"/{i.split('/')[-1].split('.tif')[0]}.jpg",img)
