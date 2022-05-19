import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import dataset_maskrcnn
from dataset_maskrcnn import MaskRCNNDataset
import torch
from references import transforms as T
from references import utils
import math
import functools
from albumentations.pytorch import ToTensorV2
import albumentations as A

def get_transform(train):
    if train:
        return A.Compose([
            # A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            ToTensorV2()
        ])

# def get_transform(train):
#     transforms = []
#     transforms.append(T.ToTensor())
#     # if train:
#     #     transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model



from references.engine import evaluate

def lr_cyclic(decay=0.2, decay_step=10, cyclic_decay=0.9, cyclic_len=25):
    def lr_step(epoch, decay, decay_step, cyclic_decay, cyclic_len):
        cyclic_n = epoch // cyclic_len
        epoch -= cyclic_len * cyclic_n
        # cyclic_decay=0.99
        # decay=0.5
        return (decay * (cyclic_decay ** cyclic_n)) ** (epoch // decay_step)

    def lr_exp(epoch, decay, decay_step, cyclic_decay, cyclic_len):
        cyclic_n = epoch // cyclic_len
        epoch -= cyclic_len * cyclic_n
        return math.exp(-decay * epoch) * (cyclic_decay ** cyclic_n)

    def lr_cos(epoch, decay, decay_step, cyclic_decay, cyclic_len):
        """ Returns alpha * cos((epoch - a) / (b - a) * pi) + beta
            a, b - left and right bounds of i-th cycle, i=0,1,...
            So lr = lr0 * lr_cos
        """
        n = 0
        k = 1
        cyclic_sum = cyclic_len
        while epoch >= cyclic_sum:
            k *= decay_step
            n += 1
            cyclic_sum += k * cyclic_len
        b = cyclic_sum
        a = b - k * cyclic_len
        alpha = 0.5 * (1 - decay)
        beta = 0.5 * (1 + decay)

        return (alpha * math.cos((epoch - a) / (b - a) * math.pi) + beta) * (cyclic_decay ** n)

    def lr_poly(epoch, decay, decay_step, cyclic_decay, cyclic_len):
        cyclic_n = epoch // cyclic_len
        epoch -= cyclic_len * cyclic_n
        return (1 - epoch / (1.048 * cyclic_len)) ** 0.9 * (cyclic_decay ** cyclic_n)

    return functools.partial(
#         lr_exp,
#         lr_poly,
        lr_cos,
        decay=decay, decay_step=decay_step,
        cyclic_decay=cyclic_decay, cyclic_len=cyclic_len
    )

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = MaskRCNNDataset('/home/ari/data/ZU/test_new_maskrcnn/dataset_10242/train', get_transform(train=True), preprocess=dataset_maskrcnn.preprocess_rgb())
    dataset_test = MaskRCNNDataset('/home/ari/data/ZU/test_new_maskrcnn/dataset_10242/test', get_transform(train=False), preprocess=dataset_maskrcnn.preprocess_rgb())

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)


    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0005, weight_decay=0.0001)


    start_epoch = 0
    # and a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lr_cyclic(
                                                  decay=0.01,
                                                  decay_step=1.2,
                                                  cyclic_decay=0.9,
                                                  cyclic_len=50),
                                                  last_epoch=start_epoch - 1)

    print('optimizer lr:', optimizer.param_groups[0]['lr'])

    # let's train it for 10 epochs
    num_epochs = 100
    print_freq = 20
    for epoch in range(num_epochs):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # for i in targets:
            #     i['masks'] = i['masks'].squeeze()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        scheduler.step()
        if epoch % 5 == 0:
            evaluate(model, data_loader_test, device=device, epoch=epoch)

    print("That's it!")


if __name__ == '__main__':
    main()
