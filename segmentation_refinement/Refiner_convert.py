import torch
import os
from  Refiner_base_model import RefinerBaseModel

### Model Paths
filePath = os.path.dirname(__file__)
modelPath = os.path.join(filePath,"pretrained/new_model.pth")
nukeModelPath = os.path.join(filePath,"nuke/cascade_psp.pt")

### Save the TorchScript model
model = RefinerBaseModel(modelPath)
model.eval()
module = torch.jit.script(model)
module.save(nukeModelPath)