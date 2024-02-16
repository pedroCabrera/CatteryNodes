import nuke
import os

def createCatModel():
    cat_file_creator = nuke.createNode('CatFileCreator')

    path = os.path.join(os.path.dirname(__file__), "nuke")

    channelsIn = "rgba.red,rgba.green,rgba.blue"
    channelsOut = "rgba.red"
    outputscale = 1

    custom_knob = nuke.Boolean_Knob("reset","reset")
    custom_knob.setValue(True)
    cat_file_creator.addKnob(custom_knob)

    custom_knob = nuke.Double_Knob("downsample_ratio","downsample_ratio")
    custom_knob.setRange(0,1)
    custom_knob.setValue(0.25)
    cat_file_creator.addKnob(custom_knob)

    for variant in ['resnet50', 'mobilenetv3']:
        model = "rvm_"+variant+".pt"
        catFile = "PC_RobustVideoMatting/rvm_"+variant+".cat"     

        cat_file_creator['torchScriptFile'].setValue(os.path.join(path,model).replace("\\", "/"))
        cat_file_creator['catFile'].setValue(os.path.join(path,catFile).replace("\\", "/"))
        cat_file_creator['channelsIn'].setValue(channelsIn)
        cat_file_creator['channelsOut'].setValue(channelsOut)
        cat_file_creator['modelId'].setValue(model.split(".")[0])
        cat_file_creator['outputScale'].setValue(outputscale)

        button_knob = cat_file_creator.knob('createCatFile')
        button_knob.execute()

