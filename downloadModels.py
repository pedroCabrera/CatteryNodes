from urllib.request import urlretrieve
import os
# Robust Video Matting Weights
url1 = (r"https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth")
url2= (r"https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth")
filename1 = "RobustVideoMatting/pretrained/rvm_mobilenetv3.pth"
filename2 = "RobustVideoMatting/pretrained/rvm_resnet50.pth"

for url, filename in zip([url1, url2], [filename1, filename2]):
    if not os.path.exists(filename):
        print(f"Downloading {filename}")  
        urlretrieve(url, filename)
        print(f"Downloaded {filename}")  

# Segment anything Weights
url1 = (r"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
url2= (r"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
url3= (r"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
filename1 = "segment-anything/pretrained/sam_vit_h.pth"
filename2 = "segment-anything/pretrained/sam_vit_l.pth"
filename3 = "segment-anything/pretrained/sam_vit_b.pth"

for url, filename in zip([url1, url2, url3], [filename1, filename2, filename3]):
    if not os.path.exists(filename):
        print(f"Downloading {filename}")  
        urlretrieve(url, filename)
        print(f"Downloaded {filename}")    