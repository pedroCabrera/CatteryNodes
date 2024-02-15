import torch
import os
from  Refiner_base_model import RefinerBaseModel

def convert(optimize_for_inference=False):
    print(f"Compiling Segmentation Refinement Model")      
    ### Model Paths
    filePath = os.path.dirname(__file__)
    modelPath = os.path.join(filePath,"pretrained/new_model.cascadepsp")
    nukeModelPath = os.path.join(filePath,"nuke/cascade_psp.pt")

    ### Save the TorchScript model
    model = RefinerBaseModel(modelPath)
    model.eval()
    module = torch.jit.script(model)
    if optimize_for_inference:
        module = torch.jit.optimize_for_inference(module)    
    module.save(nukeModelPath)
    print(f"Segmentation Refinement Model saved correctly")     
if __name__ == "__main__":
    convert()