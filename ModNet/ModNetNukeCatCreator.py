import nuke
import os

def createCatModel():
    cat_file_creator = nuke.createNode('CatFileCreator')

    path = os.path.join(os.path.dirname(__file__), "nuke")

    channelsIn = "rgba.red,rgba.green,rgba.blue"
    channelsOut = "rgba.red"
    outputscale = 1

    for variant in ['photographic_portrait_matting', 'webcam_portrait_matting']:
        model = "modnet_"+variant+".pt"
        catFile = "PC_ModNet/modnet_"+variant+".cat"     

        cat_file_creator['torchScriptFile'].setValue(os.path.join(path,model).replace("\\", "/"))
        cat_file_creator['catFile'].setValue(os.path.join(path,catFile).replace("\\", "/"))
        cat_file_creator['channelsIn'].setValue(channelsIn)
        cat_file_creator['channelsOut'].setValue(channelsOut)
        cat_file_creator['modelId'].setValue(model.split(".")[0])
        cat_file_creator['outputScale'].setValue(outputscale)

        button_knob = cat_file_creator.knob('createCatFile')
        button_knob.execute()

