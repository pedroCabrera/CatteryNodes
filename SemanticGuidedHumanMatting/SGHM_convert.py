import os
import torch
import torch.nn as nn
import sys

from SGHM_base_model import SGHMBaseModel

def convert(optimize_for_inference=False):

    filePath = os.path.dirname(__file__)

    print(f"Compiling SGHM-ResNet50")        
    ### Model Paths
    modelPath = os.path.join(filePath,"pretrained/"+"SGHM-ResNet50"+".pth")
    nukeModelPath = os.path.join(filePath,"nuke/"+"SGHM-ResNet50"+".pt")

    # create MODNet and load the pre-trained ckpt
    model = SGHMBaseModel(modelPath)
    example_input = torch.rand(1, 3, 224, 224).to(device="cuda")
    # export to TorchScript model
    model = torch.jit.trace(model,example_input)
    if optimize_for_inference:
        model = torch.jit.optimize_for_inference(model)        
    model.save(nukeModelPath)

    print(f"SGHM-ResNet50 saved correctly")  

if __name__ == "__main__":
    convert()