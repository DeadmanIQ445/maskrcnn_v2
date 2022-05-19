import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset_fasterrcnn import PennFudanDataset, TreeDataset
import torch
from references import transforms as T
from references import utils
import math
import sys
import backbones
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.rpn import AnchorGenerator

def get_RetinaNet(num_classes):
    # backbone = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True).backbone
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet101','False')
    # backbone = model
    # backbone, anchor_generator, roi_pooler = backbones.get_backbone('resnet_34')
    anchor_sizes = tuple((int(x * 0.485), int(x * 0.450), int(x * 0.750)) for x in [ 12, 24, 48, 96, 192 ])
    aspect_ratios = ((1/1.5, 1.0, 1.5),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    # anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

    model = RetinaNet(backbone,
                       num_classes=num_classes,
                       anchor_generator=anchor_generator,
                       min_size=400, max_size=400,
                       image_mean=(0,0,0), image_std=(1,1,1),
                       detections_per_img=200,
                       # topk_candidates=3000
                      )
    # cls_logits = torch.nn.Conv2d(out_channels, len(anchor_sizes) * num_classes, kernel_size=3, stride=1, padding=1)
    # torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    # torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
    # # assign cls head to model
    # model.head.classification_head.cls_logits = cls_logits
    return model


def get_fasterrcnn(num_classes):

    # backbone, anchor_generator, roi_pooler = backbones.get_backbone('resnet_101')
    # sizes = ((8, 16, 32, 64, 128),)
    # aspect_ratios = ((0.75, 1.0, 1.25),)
    # anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

    anchor_sizes = tuple((int(x * 0.485), int(x * 0.450), int(x * 0.750)) for x in [ 16, 32, 64, 128, 256 ])
    aspect_ratios = ((1/1.5, 1.0, 1.5),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # backbone = model.backbone

    model = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet152','True')
    backbone = model


    model = backbones.FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       # rpn_pre_nms_top_n_train=4000, rpn_pre_nms_top_n_test=2000,
                       # rpn_post_nms_top_n_train=4000, rpn_post_nms_top_n_test=2000,
                       box_detections_per_img = 200,
                       # box_roi_pool=roi_pooler,
                       box_nms_thresh=0.3,
                       rpn_nms_thresh=0.3,
                       image_mean=[0,0,0], image_std=[1,1,1], min_size=400)

    #

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    return model

from references.engine import evaluate
import os

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    num_classes = 2

    samples_dir = 'summer_tiles_200'

    lr = 0.00005

    dataset = TreeDataset(csv_file=os.path.join(samples_dir,"train_val.csv"),root_dir=samples_dir)
    dataset_test = TreeDataset(csv_file=os.path.join(samples_dir,"test.csv"),root_dir=samples_dir, train=False)


    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    model = get_RetinaNet(num_classes)
    # model = get_fasterrcnn(num_classes)
    model = torch.load('./models/RetinaNet_0.6219756949905441_95.pth')

    model.to(device)
    save_dir= "models/exp2"

    optimizer = torch.optim.Adam(model.parameters(),
                           lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=50)

    # let's train it for 10 epochs
    num_epochs = 300
    print_freq = 1000
    best_map = 0
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for images, targets, image_name in metric_logger.log_every(data_loader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()


            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        scheduler.step()
        if epoch%5 == 0:
            eval_ret = evaluate(model, data_loader_test, device=device, epoch=epoch)
            if eval_ret.coco_eval['bbox'].stats[1] > best_map:
                best_map = eval_ret.coco_eval['bbox'].stats[1]
                torch.save(model, f"{save_dir}/{model.__class__.__name__}_{best_map}_{epoch}.pth")

        

    print("That's it!")


if __name__ == '__main__':
    main()
