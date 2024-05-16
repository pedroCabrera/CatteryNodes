# CatteryNodes
Collection of AI models to work with Nuke Cattery system for fast inference

# Install
1. Go to [Releases](https://github.com/pedroCabrera/CatteryNodes/releases) and download the file Cattery.rar
2. Extract into a Cattery folder and copy into your .nuke or your plugins path.

# MODELS:
## Modnet:
https://github.com/ZHKKKe/MODNet
## Robust Video Matting:
https://github.com/PeterL1n/RobustVideoMatting
## Semantic Guided Human Matting - SGHM
https://github.com/cxgincsu/SemanticGuidedHumanMatting
## Segment Anything ( only base and large, huge not yet suported by nuke):
https://github.com/facebookresearch/segment-anything
## Cascade PSP (segmentation Refinement ):
https://github.com/hkchengrex/CascadePSP

# Compiling
There are a few utility files to download and to build all the models and also to create the corresponding Cattery files from the downloaded models as well as to copy them to your user folder to fast test.

Each model has a torchscript compilable version if the original one is not, I tried to preserve the original repos intact and just mock it or change minor stuff to make it work.
