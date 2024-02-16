import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "MODNet"))

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from ModNet_base_model import MODNetBaseModel

from PIL import Image
import numpy as np


def ensure_divisible_by_32(tensor):
    _, _, height, width = tensor.shape
    new_height = ((height - 1) // 32 + 1) * 32
    new_width = ((width - 1) // 32 + 1) * 32

    pad_height = max(new_height - height, 0)
    pad_width = max(new_width - width, 0)

    # Padding format: (left, right, top, bottom)
    padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
    tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)

    return tensor

def display_tensor_image(tensor):
    # Remove the batch dimension and squeeze the channel dimension
    tensor = tensor.squeeze(0).squeeze(0)
    
    # Convert the tensor to a PIL Image
    transform = transforms.ToPILImage()
    image = transform(tensor)

    # Display the image
    image.show()


def split_image(input_tensor, square_size):
    """
    Split the input tensor into square images of a given size.
    Args:
    - input_tensor: Input tensor of shape [1, channels, width, height]
    - square_size: Size of the square images
    Returns:
    - List of square images
    """
    _, channels, width, height = input_tensor.shape
    input_image = input_tensor.squeeze(0)
    
    # Calculate number of rows and columns of squares
    num_rows = height // square_size
    num_cols = width // square_size
    
    squares = []
    for r in range(num_rows):
        for c in range(num_cols):
            square = input_image[:, 
                                 c * square_size:(c + 1) * square_size, 
                                 r * square_size:(r + 1) * square_size]
            squares.append(square.unsqueeze(0))
    
    # If there are remaining columns, create a square using them
    if width % square_size != 0:
        for r in range(num_rows):
            square = input_image[:, 
                                 num_cols * square_size:width, 
                                 r * square_size:(r + 1) * square_size]
            squares.append(square.unsqueeze(0))
    
    # If there are remaining rows, create squares using them
    if height % square_size != 0:
        for c in range(num_cols):
            square = input_image[:, 
                                 c * square_size:(c + 1) * square_size, 
                                 num_rows * square_size:height]
            squares.append(square.unsqueeze(0))
    
    # If there are remaining rows and columns, create a square using them
    if width % square_size != 0 and height % square_size != 0:
        square = input_image[:, 
                             num_cols * square_size:width, 
                             num_rows * square_size:height]
        squares.append(square.unsqueeze(0))
    
    return squares

def merge_images(squares, original_size):
    """
    Merge the list of square images to form the original image.
    Args:
    - squares: List of square images
    - original_size: Original size of the image
    Returns:
    - Merged image tensor of shape [1, channels, width, height]
    """
    _, channels, width, height = original_size
    output_image = torch.zeros(original_size)
    
    idx = 0
    for r in range(height // squares[0].shape[2]):
        for c in range(width // squares[0].shape[3]):
            output_image[:, :, 
                          c * squares[0].shape[3]:(c + 1) * squares[0].shape[3],
                          r * squares[0].shape[2]:(r + 1) * squares[0].shape[2]] = squares[idx]
            idx += 1
    
    # If there are remaining columns
    if width % squares[0].shape[3] != 0:
        for r in range(height // squares[0].shape[2]):
            output_image[:, :, 
                          width - (width % squares[0].shape[3]):width,
                          r * squares[0].shape[2]:(r + 1) * squares[0].shape[2]] = squares[idx]
            idx += 1
    
    # If there are remaining rows
    if height % squares[0].shape[2] != 0:
        for c in range(width // squares[0].shape[3]):
            output_image[:, :, 
                          c * squares[0].shape[3]:(c + 1) * squares[0].shape[3],
                          height - (height % squares[0].shape[2]):height] = squares[idx]
            idx += 1
    
    # If there are remaining rows and columns
    if width % squares[0].shape[3] != 0 and height % squares[0].shape[2] != 0:
        output_image[:, :, 
                      width - (width % squares[0].shape[3]):width,
                      height - (height % squares[0].shape[2]):height] = squares[idx]
    
    return output_image





filePath = os.path.dirname(__file__)
modelPath = os.path.join(filePath,"pretrained/modnet_photographic_portrait_matting"+".ckpt")
print(modelPath)
imagePath = os.path.join(filePath,"test_image_internet2.jpg")

### Load image and mask
raw_image = Image.open(imagePath).convert("RGB")
raw_image = np.array(raw_image).astype(np.float32)/255.0  
torch_image = torch.from_numpy(raw_image).to(device="cuda")
torch_image = torch_image.permute(2, 0, 1)[None,:]
print(torch_image.shape)
torch_image = ensure_divisible_by_32(torch_image)
print(torch_image.shape)

### Run the model for testing
# create MODNet and load the pre-trained ckpt
modnet = MODNetBaseModel(backbone_pretrained=False)
modnet = nn.DataParallel(modnet).cuda()
state_dict = torch.load(modelPath)
modnet.load_state_dict(state_dict)
modnet.eval()
patches = split_image(torch_image,512)
masks = []


for patch in patches:
    mask = modnet(patch)
    #print(mask.shape)
    #display_tensor_image(patch)
    masks.append(mask)

merged_image = merge_images(masks, torch_image.shape)
display_tensor_image(merged_image)
print(merged_image.shape)

"""
mask = modnet((torch_image))
print(mask.shape)
display_tensor_image(mask)
"""
  
