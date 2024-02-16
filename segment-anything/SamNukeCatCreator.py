import nuke
import os

def createCatModel():
    cat_file_creator = nuke.createNode('CatFileCreator')

    path = os.path.join(os.path.dirname(__file__), "nuke")
    channelsIn = "rgba.red,rgba.green,rgba.blue"
    channelsOut = "rgba.red,rgba.green,rgba.blue"
    outputscale = 1

    for i,pos in enumerate(["p1", "p2", "p3", "p4","p5","n1","n2","n3","n4","n5"]):
        custom_knob = nuke.XY_Knob(pos,pos)
        y = 0 if i < 5 else 100
        custom_knob.setValue(((i%5)+1)*-100,0)
        custom_knob.setValue(y,1)
        if i == 0:
            custom_knob.setValue(100,0)
            custom_knob.setValue(100,1)
        cat_file_creator.addKnob(custom_knob)

    for variant in ['vit_b', 'vit_l',]: # 'vit_h' actually creashes Nuke
        model = "sam_"+variant+".pt"
        catFile = "PC_Segment_anything/sam_"+variant+".cat"     

        cat_file_creator['torchScriptFile'].setValue(os.path.join(path,model).replace("\\", "/"))
        cat_file_creator['catFile'].setValue(os.path.join(path,catFile).replace("\\", "/"))
        cat_file_creator['channelsIn'].setValue(channelsIn)
        cat_file_creator['channelsOut'].setValue(channelsOut)
        cat_file_creator['modelId'].setValue(model.split(".")[0])
        cat_file_creator['outputScale'].setValue(outputscale)

        button_knob = cat_file_creator.knob('createCatFile')
        button_knob.execute()

