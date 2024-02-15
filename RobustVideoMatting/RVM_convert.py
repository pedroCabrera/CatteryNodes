import os
import torch
from RVM_base_model_rec import RvmBaseModel

def convert(optimize_for_inference=False):

    filePath = os.path.dirname(__file__)
    for variant in ['resnet50', 'mobilenetv3']:
        print(f"Compiling RobustVideoMate Variant {variant}")        
        ### Model Paths
        modelPath = os.path.join(filePath,"pretrained/rvm_"+variant+".pth")
        nukeModelPath = os.path.join(filePath,"nuke/rvm_"+variant+".pt")

        ### Save the TorchScript model
        model = RvmBaseModel(variant, modelPath)
        module = torch.jit.script(model)
        if optimize_for_inference:
            module = torch.jit.optimize_for_inference(module)
        module.save(nukeModelPath)    
        print(f"RobustVideoMate Variant {variant} saved correctly")  

if __name__ == "__main__":
    convert()