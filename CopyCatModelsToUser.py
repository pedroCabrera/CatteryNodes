import os
import sys
import shutil

currPath = os.path.dirname(__file__)

robustVideoMateCatCompiler = os.path.join(currPath, "RobustVideoMatting", "nuke", "PC_RobustVideoMatting" )
segment_anythingCompiler = os.path.join(currPath, "segment-anything", "nuke", "PC_Segment_anything" )
segmentationRefinementCatCompiler = os.path.join(currPath, "segmentation_refinement", "nuke", "PC_Segmentation_Refinement")
SGHM_CatCompiler = os.path.join(currPath, "SemanticGuidedHumanMatting", "nuke", "PC_SemanticGuidedHumanMatting")
MODNetCatCompiler = os.path.join(currPath, "ModNet", "nuke", "PC_Improved_ModNet")

for covneter in [robustVideoMateCatCompiler, segment_anythingCompiler, segmentationRefinementCatCompiler,
                 SGHM_CatCompiler, MODNetCatCompiler]:
    folderName = os.path.basename(covneter)
    destination_dir = os.path.join(os.path.expanduser('~'),".nuke","Cattery",folderName)
    
    shutil.copytree(covneter, destination_dir, dirs_exist_ok=True)