import sys
import os

import torch
from  Refiner_base_model import RefinerBaseModel

from PIL import Image
import numpy as np




filePath = os.path.dirname(__file__)
modelPath = os.path.join(filePath,"pretrained/new_model.cascadepsp")
nukeModelPath = os.path.join(filePath,"pretrained/nuke/cascade_psp_v3.pt")

imagePath = os.path.join(filePath,"truck.jpg")
maskPath = os.path.join(filePath,"truck_mate.jpg")

### Load image and mask
raw_image = Image.open(imagePath).convert("RGB")
raw_image = np.array(raw_image).astype(np.float32)/255.0  
torch_image = torch.from_numpy(raw_image).to(device="cuda")
torch_image = torch_image.permute(2, 0, 1)[None,:]

raw_mask = Image.open(maskPath).convert('L')
raw_mask = np.array(raw_mask).astype(np.float32)/255.0  
torch_mask = torch.from_numpy(raw_mask).to(device="cuda")
torch_mask = torch_mask.unsqueeze(0)  # Add a dimension for batch size

torch_image_mask = torch.cat((torch_image, torch_mask.unsqueeze(1)), dim=1)

### Run the model for testing
model = RefinerBaseModel(modelPath).to(device="cuda")
module = torch.jit.script(model)

mask = module((torch_image_mask))

  
