import os

import numpy as np
import torch

from parameters import ROOT_DIR
from tests.test_train_class import VideoClassifier

classes = ['115x200', '115x400', '150x300', '60x90', '85x150']
# dataset = 128 * int(input_size[1] / 16) * int(input_size[2] / 16) * input_size[0],.create_box_video_dataset(
#         box_path=os.path.join(ROOT_DIR, 'tests/class_boxes_26_model3_full.dict'),
#         split=0.9,
#         frame_size=(128, 128),
# )
model = VideoClassifier(weights='/media/deny/Новый том/AI/CarpetTracker/video_class_train/video data_94%/best.pt')


def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output

print(model.model.code)
xxx = nested_children(model.model)
for name, child in xxx.items():
    if 'conv3d' in name:
        print(f"name={name}, in_channels={child.in_channels}, out_channels={child.out_channels}, kernel_size={child.kernel_size}")
    if 'dense' in name:
        print(f"name={name}, in_features={child.in_features}, out_features={child.out_features}")
