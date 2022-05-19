import torchvision
from torch import nn

def get_backbone(backbone_name):
    if backbone_name == 'resnet_18':
        resnet_net = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 512
    elif backbone_name == 'resnet_34':
        resnet_net = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 512
    elif backbone_name == 'resnet_50':
        resnet_net = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == 'resnet_101':
        resnet_net = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == 'resnet_152':
        resnet_net = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == 'resnext101_32x8d':
        resnet_net = torchvision.models.resnext101_32x8d(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    else:
        resnet_net = torchvision.models.mobilenet_v2(pretrained=True).features
        modules = list(resnet_net.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    return backbone, anchor_generator, roi_pooler
def another(backbone_name):
    if backbone_name == 'resnet_18':
            resnet_net = torchvision.models.resnet18(pretrained=True)
            modules = list(resnet_net.children())[:-2]
            backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnet_34':
        resnet_net = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnet_50':
        resnet_net = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnet_101':
        resnet_net = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnet_152':
        resnet_net = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnet_50_modified_stride_1':
        resnet_net = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnext101_32x8d':
        resnet_net = torchvision.models.resnext101_32x8d(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
    return backbone

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

if __name__=="__main__":
    backbone, anchor_generator, roi_pooler = get_backbone('resnet_18')


    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
