import os
import torch

from patches import apply_patches
from sam_predictor_base_model import SamPredictorBaseModel

def convert(optimize_for_inference=False):
    ### Apply patches to fix some issues with the SAM model and TorchScript
    apply_patches()

    filePath = os.path.dirname(__file__)

    for variant in ['vit_b', 'vit_l', 'vit_h']:
        print(f"Compiling Sam Variant {variant}") 
        ### Model Paths
        modelPath = os.path.join(filePath,"pretrained", "sam_" + variant +".pth")
        nukeModelPath = os.path.join(filePath,"nuke", "sam_" + variant +".pt")

        model = SamPredictorBaseModel(variant=variant, modelPath=modelPath)
        model.eval()

        model = torch.jit.script(model)
        
        # This is commented as it does work work with the SAM model.
        if optimize_for_inference:
            model = torch.jit.optimize_for_inference(model)

        ### Save the TorchScript model
        model.save(nukeModelPath)
        print(f"Sam Variant {variant} saved correctly") 

if __name__ == "__main__":
    convert()

