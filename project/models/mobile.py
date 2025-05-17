import torch
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

class LightFasterRCNNMobileNetV3(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Load MobileNetV3 backbone
        backbone = mobilenet_v3_large(weights=None).features

        # Define FPN layers to extract
        return_layers = {'5': '0', '9': '1'}  # Layer 5 (40 channels), Layer 9 (80 channels)
        in_channels_list = [40, 80]
        fpn_out_channels = 256

        # Build FPN backbone
        self.backbone_with_fpn = BackboneWithFPN(
            backbone,
            return_layers=return_layers,
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
            extra_blocks=None  # No LastLevelMaxPool
        )

        # Anchor Generator
        self.anchor_generator = AnchorGenerator(
            sizes=((60, 90),) * 3,
            aspect_ratios=((0.67, 1.0, 1.5),) * 3
        )

        # ROI Box Head
        representation_size = 640
        self.box_head = TwoMLPHead(
            in_channels=fpn_out_channels * 7 * 7,
            representation_size=representation_size
        )

        # ROI Box Predictor
        self.box_predictor = FastRCNNPredictor(
            in_channels=representation_size,
            num_classes=num_classes
        )

        # Build the full model
        self.model = FasterRCNN(
            backbone=self.backbone_with_fpn,
            num_classes=None,  # We use custom box predictor
            rpn_anchor_generator=self.anchor_generator,
            box_head=self.box_head,
            box_predictor=self.box_predictor,
            rpn_pre_nms_top_n_train=300,
            rpn_pre_nms_top_n_test=150,
            rpn_post_nms_top_n_train=200,
            rpn_post_nms_top_n_test=100,
        )

        # Make backbone trainable
        for param in self.model.backbone.parameters():
            param.requires_grad = True

        self._print_param_summary()

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def _count_parameters(self, module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def _print_param_summary(self):
        print("Total parameters:        ", self._count_parameters(self.model))
        print("Backbone (MobileNet):    ", self._count_parameters(self.model.backbone))
        print("RPN:                     ", self._count_parameters(self.model.rpn))
        print("ROI Heads (Box Head):    ", self._count_parameters(self.model.roi_heads))