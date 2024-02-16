import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "MODNet"))

import torch
import torch.nn as nn

from MODNet.torchscript.modnet_torchscript import IBNorm, Conv2dIBNormRelu, SEBlock, LRBranch, HRBranch, FusionBranch
from src.models.backbones import SUPPORTED_BACKBONES

class MODNetBaseModel(nn.Module):
    """ Architecture of MODNetBaseModel
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=True):
        super(MODNetBaseModel, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()                

    def forward(self, img):
        # NOTE
        dtype = img.dtype
        if(img.is_cuda):
           device = torch.device('cuda')
        else:
           device = torch.device('cpu')   
        src = img.to(device, dtype, non_blocking=True)  
        #self.to(device, dtype, non_blocking=True)     
        mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(device, dtype, non_blocking=True)
        std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(device, dtype, non_blocking=True)
        src =  (src - mean) / std        
        lr_out = self.lr_branch(src)
        lr8x = lr_out[0]
        enc2x = lr_out[1]
        enc4x = lr_out[2]

        hr2x = self.hr_branch(src, enc2x, enc4x, lr8x)
        
        pred_matte = self.f_branch(src, lr8x, hr2x)

        return pred_matte
    
    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)

