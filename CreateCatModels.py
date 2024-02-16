import os
import sys
import importlib
currPath = os.path.dirname(__file__)

robustVideoMateCatCompiler = os.path.join(currPath, "RobustVideoMatting", "RVMNukeCatCreator.py")
segment_anythingCompiler = os.path.join(currPath, "segment-anything", "SamNukeCatCreator.py")
segmentationRefinementCatCompiler = os.path.join(currPath, "segmentation_refinement", "RefinerNukeCatCreator.py")
MODNetCatCompiler = os.path.join(currPath, "ModNet", "ModNetNukeCatCreator.py")
SGHM_CatCompiler = os.path.join(currPath, "SemanticGuidedHumanMatting", "SGHMNukeCatCreator.py")

for covneter in [robustVideoMateCatCompiler, segment_anythingCompiler, segmentationRefinementCatCompiler,
                 MODNetCatCompiler, SGHM_CatCompiler]:
    sys.path.append(os.path.dirname(covneter))
    mod = importlib.import_module(os.path.basename(covneter).replace(".py", ""))
    met = getattr(mod, "createCatModel")
    met()
