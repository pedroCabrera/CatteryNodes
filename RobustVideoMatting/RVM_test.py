import torch
from RobustVideoMatting.model import MattingNetwork

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from RobustVideoMatting.inference_utils import VideoReader, VideoWriter

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load(r"C:\Users\pedro\GoogleDrive\Mi unidad\CODE\AI\Nuke\Cattery\RobustVideoMatting\pretrained\rvm_mobilenetv3.pth"))



reader = VideoReader(r'C:\Users\pedro\Desktop\TEST_FILES\VIDEO\ASU_104_022_0030_paraRVM_v003.mp4', transform=ToTensor())
writer = VideoWriter(r"C:\Users\pedro\Desktop\TEST_FILES\VIDEO\ASU_104_022_0030_RVM_multiFrame_v003.mp4", frame_rate=30, bit_rate = 100000000)
writer2 = VideoWriter(r"C:\Users\pedro\Desktop\TEST_FILES\VIDEO\ASU_104_022_0030_RVM_simple_v003.mp4", frame_rate=30, bit_rate = 100000000)

bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
rec = [None] * 4                                       # Initial recurrent states.
downsample_ratio = 0.5                                # Adjust based on your video.
i=0
with torch.no_grad():
    for src in DataLoader(reader):# RGB tensor normalized to 0 ~ 1
        fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.
        com = fgr * pha
        writer.write(com)                              # Write frame.P       

        fgr, pha = model(src.cuda(), downsample_ratio=downsample_ratio)[:2]  # Cycle the recurrent states.
        com = fgr * pha
        writer2.write(com)                              # Write frame.P       
        i+=1        