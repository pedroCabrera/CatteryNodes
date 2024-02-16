import os
import sys
import importlib
robustVideoMateConverter = os.path.join(os.path.dirname(__file__), "RobustVideoMatting", "RVM_convert.py")
segment_anythingCoverter = os.path.join(os.path.dirname(__file__), "segment-anything", "sam_convert.py")
segmentationRefinementConverter = os.path.join(os.path.dirname(__file__), "segmentation_refinement", "Refiner_convert.py")
MODNetConverter = os.path.join(os.path.dirname(__file__), "ModNet", "ModNet_convert.py")

for covneter in [robustVideoMateConverter, segment_anythingCoverter, segmentationRefinementConverter,
                 MODNetConverter ]:
    sys.path.append(os.path.dirname(covneter))
    mod = importlib.import_module(os.path.basename(covneter).replace(".py", ""))
    met = getattr(mod, "convert")
    met()
