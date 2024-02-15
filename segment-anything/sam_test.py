import os
import torch
from segment_anything import sam_model_registry , SamPredictor
from patches import apply_patches
from sam_predictor_base_model import SamPredictorBaseModel
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

### Apply patches to fix some issues with the SAM model and TorchScript
apply_patches()


filePath = os.path.dirname(__file__)

# An example input you would normally provide to your model's forward() method.
imagePath = os.path.join(filePath,"truck.jpg")
raw_image = Image.open(imagePath).convert("RGB")

raw_image = np.array(raw_image).astype(float)/255.0  
torch_image = torch.from_numpy(raw_image)
torch_image = torch_image.permute(2, 0, 1)[None,:]

device = "cuda"


for variant in ['vit_b', 'vit_l', 'vit_h']:
    ### Model Paths
    modelPath = os.path.join(filePath,"pretrained", "sam_" + variant +".pth")
    nukeModelPath = os.path.join(filePath,"nuke", "sam_" + variant +".pt")
    
    
    # An instance of the model.
    base_model = sam_model_registry[variant](checkpoint=modelPath)
    base_model.to(device=device)



    input_point = np.array([500, 375])
    input_label = np.array([1])
    
    model = SamPredictorBaseModel(sam_model=base_model,position=torch.from_numpy(input_point))
    model.to(device=device)
    model.eval()

    example_inputs = (torch_image.to(device).float(),)

    # Preview the TorchScript model
    mask = model(*example_inputs)

    plt.figure(figsize=(10,10))
    plt.imshow(mask.cpu().squeeze().numpy().transpose(1,2,0))
    plt.axis('off')
    plt.show() 


