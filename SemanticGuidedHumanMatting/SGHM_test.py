import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "MODNet"))

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from SGHM_base_model import SGHMBaseModel

from PIL import Image
import numpy as np

def display_tensor_image(tensor):
    # Remove the batch dimension and squeeze the channel dimension
    tensor = tensor.squeeze(0).squeeze(0)
    
    # Convert the tensor to a PIL Image
    transform = transforms.ToPILImage()
    image = transform(tensor)

    # Display the image
    image.show()

filePath = os.path.dirname(__file__)
modelPath = os.path.join(filePath,"pretrained/SGHM-ResNet50.pth")
print(modelPath)
imagePath = r"C:\Users\pedro\Desktop\TEST_FILES\fotos\rata.png"

### Load image and mask
raw_image = Image.open(imagePath).convert("RGB")
raw_image = np.array(raw_image).astype(np.float32)/255.0  
torch_image = torch.from_numpy(raw_image).to(device="cuda")
torch_image = torch_image.permute(2, 0, 1)[None,:]
print(torch_image.shape)

### Run the model for testing
# create MODNet and load the pre-trained ckpt
modnet = SGHMBaseModel(modelPath)
mask = modnet((torch_image))
display_tensor_image(mask)
print(mask.shape)

"""
mask = modnet((torch_image))
print(mask.shape)
display_tensor_image(mask)
"""
  
