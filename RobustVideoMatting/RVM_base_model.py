import torch
import torch.nn as nn
from RobustVideoMatting.model import MattingNetwork

class RvmBaseModel(torch.nn.Module):
	"""
	This class is a wrapper for our custom ResNet model.
	"""
	def __init__(self, variant, modelPath, downsample_ratio = 1.0):

		super(RvmBaseModel, self).__init__()

		self.model =  MattingNetwork(variant=variant)
		self.model.load_state_dict(torch.load(modelPath))
		self.model.eval()
		self.downsample_ratio = downsample_ratio

	def forward(self, input):
		"""
		:param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
		:return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
		"""
		dtype = input.dtype
		if(input.is_cuda):
		   device = torch.device('cuda')
		else:
		   device = torch.device('cpu')     
		with torch.no_grad():	
			src = input.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
			fgr, pha = self.model(src, downsample_ratio=self.downsample_ratio)[:2]
		
		return pha[0]