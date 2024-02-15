import os
import torch
from RVM_base_model_rec import RvmBaseModel

filePath = os.path.dirname(__file__)

for variant in ['resnet50', 'mobilenetv3']:
    ### Model Paths
    modelPath = os.path.join(filePath,"pretrained/rvm_"+variant+".pth")
    nukeModelPath = os.path.join(filePath,"nuke/rvm_"+variant+".pt")

    ### Save the TorchScript model
    model = RvmBaseModel(variant, modelPath)
    module = torch.jit.script(model)
    module.save(nukeModelPath)    
