import os
import torch
import torch.nn as nn

from ModNet_base_model import MODNetBaseModel

def convert(optimize_for_inference=False):

    filePath = os.path.dirname(__file__)
    for variant in ['modnet_photographic_portrait_matting','modnet_webcam_portrait_matting']:
        print(f"Compiling MODNet Variant {variant}")        
        ### Model Paths
        modelPath = os.path.join(filePath,"pretrained/"+variant+".ckpt")
        nukeModelPath = os.path.join(filePath,"nuke/"+variant+".pt")

        # create MODNet and load the pre-trained ckpt
        modnet = MODNetBaseModel(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)#.cuda()
        state_dict = torch.load(modelPath)
        modnet.load_state_dict(state_dict)
        modnet.eval()

        # export to TorchScript model
        model = torch.jit.script(modnet.module)
        if optimize_for_inference:
            model = torch.jit.optimize_for_inference(model)        
        model.save(nukeModelPath)
 
        print(f"MODNet Variant {variant} saved correctly")  

if __name__ == "__main__":
    convert()