import torch
import torch.nn as nn
from RobustVideoMatting.model import MattingNetwork

class RvmBaseModel(torch.nn.Module):
	def __init__(self, variant, modelPath, reset = 1, downsample_ratio = 1.0):
		super(RvmBaseModel, self).__init__()

		self.model =  MattingNetwork(variant=variant)
		self.model.load_state_dict(torch.load(modelPath))
		self.model.eval()
		self.downsample_ratio = downsample_ratio
		self.reset = reset
		self.r1: torch.Tensor=torch.empty((0,))
		self.r2: torch.Tensor=torch.empty((0,))
		self.r3: torch.Tensor=torch.empty((0,))
		self.r4: torch.Tensor=torch.empty((0,))

	def forward(self, input):
		dtype = input.dtype
		if(input.is_cuda):
		   device = torch.device('cuda')
		else:
		   device = torch.device('cpu')
		if(self.reset):
			self.r1 = torch.empty((0,))
			self.r2 = torch.empty((0,))
			self.r3 = torch.empty((0,))
			self.r4 = torch.empty((0,))
		with torch.no_grad():    
			src = input.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
			self.r1 = self.r1.to(device, dtype, non_blocking=True)
			self.r2 = self.r2.to(device, dtype, non_blocking=True)
			self.r3 = self.r3.to(device, dtype, non_blocking=True)
			self.r4 = self.r4.to(device, dtype, non_blocking=True)
			if(self.reset):		
				fgr, pha, self.r1,self.r2,self.r3,self.r4, = self.model(src, downsample_ratio=self.downsample_ratio)
			else:
				fgr, pha, self.r1,self.r2,self.r3,self.r4, = self.model(src,self.r1,self.r2,self.r3,self.r4, downsample_ratio=self.downsample_ratio)
		
		return pha[0]