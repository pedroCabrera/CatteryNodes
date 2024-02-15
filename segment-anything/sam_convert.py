import os
import torch

from patches import apply_patches
from sam_predictor_base_model import SamPredictorBaseModel

from PIL import Image
import numpy as np

### Apply patches to fix some issues with the SAM model and TorchScript
apply_patches()

filePath = os.path.dirname(__file__)

for variant in ['vit_b', 'vit_l', 'vit_h']:
    ### Model Paths
    modelPath = os.path.join(filePath,"pretrained", "sam_" + variant +".pth")
    nukeModelPath = os.path.join(filePath,"nuke", "sam_" + variant +".pt")

    model = SamPredictorBaseModel(variant=variant, modelPath=modelPath)
    model.eval()

    model = torch.jit.script(model)
    
    # This is commented as it does work work with the SAM model.
    # model = torch.jit.optimize_for_inference(model)

    ### Save the TorchScript model
    model.save(nukeModelPath)

    

