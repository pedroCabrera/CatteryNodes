import nuke
import os

def createCatModel():
    cat_file_creator = nuke.createNode('CatFileCreator')

    path = os.path.join(os.path.dirname(__file__), "nuke")
    model = "cascade_psp.pt"
    catFile = r"PC_Segmentation_Refinement\cascade_psp.cat"

    channelsIn = "rgba.red,rgba.green,rgba.blue,rgba.alpha"
    channelsOut = "rgba.red"
    outputscale = 1

    cat_file_creator['torchScriptFile'].setValue(os.path.join(path,model).replace("\\", "/"))
    cat_file_creator['catFile'].setValue(os.path.join(path,catFile).replace("\\", "/"))
    cat_file_creator['channelsIn'].setValue(channelsIn)
    cat_file_creator['channelsOut'].setValue(channelsOut)
    cat_file_creator['modelId'].setValue(model.split(".")[0])
    cat_file_creator['outputScale'].setValue(outputscale)

    custom_knob = nuke.Boolean_Knob("optimize_speed","optimize_speed")
    custom_knob.setValue(False)
    cat_file_creator.addKnob(custom_knob)

    custom_knob = nuke.Int_Knob("L","L")
    custom_knob.setValue(900)
    cat_file_creator.addKnob(custom_knob)


    button_knob = cat_file_creator.knob('createCatFile')

    button_knob.execute()

#nuke.scriptClear()
#nuke.scriptExit()