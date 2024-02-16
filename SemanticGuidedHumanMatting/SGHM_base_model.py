import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "SemanticGuidedHumanMatting"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import HumanSegment, HumanMatting
# --------------- Main ---------------

infer_size = 1280
class SGHMBaseModel(nn.Module):
    """ Architecture of SGHMBaseModel
    """

    def __init__(self, modelPath):
        super(SGHMBaseModel, self).__init__()

        # Load Model
        #self.model = model
        state_dict = torch.load(modelPath)
        new_state_dict = {}
        for k, v in state_dict.items():
            index = k.find('module.')
            if index != -1:
                new_key = k[:index] + k[index+len('module.'):]
            else:
                new_key = k
            new_state_dict[new_key] = v        
        self.model = HumanMatting(backbone='resnet50')
        #self.model = nn.DataParallel(self.model).eval()
        self.model.load_state_dict(new_state_dict)   
        self.model.cuda().eval()
        

    def forward(self, img):
        dtype = img.dtype
        if(img.is_cuda):
           device = torch.device('cuda')
           self.model.cuda()
        else:
           device = torch.device('cpu')   
        
        B, C, h, w = img.shape
        if w >= h:
            rh = infer_size
            rw = int(w / h * infer_size)
        else:
            rw = infer_size
            rh = int(h / w * infer_size)
        rh = rh - rh % 64
        rw = rw - rw % 64    
        img = img.to(device, dtype, non_blocking=True)  

        input_tensor = F.interpolate(img, size=(rh, rw), mode='bilinear')
        with torch.no_grad():
            pred = self.model(input_tensor)

        # progressive refine alpha
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        pred_alpha = alpha_pred_os8.clone().detach()
        weight_os4 = self.get_unknown_tensor_from_pred(pred_alpha, rand_width=30, train_mode=False)
        pred_alpha[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = self.get_unknown_tensor_from_pred(pred_alpha, rand_width=15, train_mode=False)
        pred_alpha[weight_os1>0] = alpha_pred_os1[weight_os1>0]

        #pred_alpha = pred_alpha.repeat(1, 3, 1, 1)
        pred_alpha = F.interpolate(pred_alpha, size=(h, w), mode='bilinear')

        # output segment
        #pred_segment = pred['segment']
        #pred_segment = F.interpolate(pred_segment, size=(h, w), mode='bilinear')

        return pred_alpha

    def get_unknown_tensor_from_pred(self, pred, rand_width=30, train_mode=True):
        ### pred: N, 1 ,H, W 
        N, C, H, W = pred.shape

        uncertain_area = torch.ones_like(pred)
        uncertain_area[pred<1.0/255.0] = 0
        uncertain_area[pred>1-1.0/255.0] = 0

        for n in range(N):
            uncertain_area_ = uncertain_area[n,0,:,:] # H, W
            if train_mode:
                width = torch.randint(1, rand_width, (1,)).item()
            else:
                width = rand_width // 2
            padding = width // 2
            uncertain_area_ = F.max_pool2d(uncertain_area_.unsqueeze(0).unsqueeze(0), kernel_size=width, stride=1, padding=padding).squeeze(0).squeeze(0)
            uncertain_area[n,0,:,:] = uncertain_area_

        weight = torch.zeros_like(uncertain_area)
        weight[uncertain_area == 1] = 1

        return weight.cuda()