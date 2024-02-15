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

def main():
    """
    Convert SAM to TorchScript and save it.

    See: http://docs.djl.ai/docs/pytorch/how_to_convert_your_model_to_torchscript.html
    """
    apply_patches()

    device = "cuda"
    # An instance of the model.
    base_model = sam_model_registry["vit_l"](checkpoint="G:/Mi unidad/CODE/AI/segment-anything/pretrained/sam_vit_l_0b3195.pth")
    base_model.to(device=device)

    predictor = SamPredictor(base_model)
    #model = SamPredictorBaseModel(model=base_model)
    raw_image = Image.open("F:/descargas/truck.jpg").convert("RGB")
    raw_image = np.array(raw_image)
    
    print(raw_image)
    print(raw_image.shape)

    # Define a transformation to convert the PIL image to a torch tensor
    #transform = transforms.ToTensor()

    ## Apply the transformation to get the torch tensor
    #torch_tensor = transform(raw_image)
    #torch_tensor_uint8 = (torch_tensor * 255).to(torch.uint8)#.to(device=device)
    torchImage = predictor.set_image(raw_image)
    print(torchImage)
    print(torchImage.shape)   
    
    input_point = np.array([[500, 375]])
    
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )    
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        print(mask)
        plt.figure(figsize=(10,10))
        plt.imshow(raw_image)
        maskImage = show_mask(mask, plt.gca())
        print(maskImage)
        print(maskImage.shape)
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show() 


if __name__ == "__main__":
    main()
