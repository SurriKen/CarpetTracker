import os

import cv2
import numpy as np
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes

from parameters import ROOT_DIR
from tests.test_dataset import clean_diff_image
from utils import load_txt

vid1 = os.path.join(ROOT_DIR, 'datasets/class_videos/60x90/camera_1/7.mp4')
boxes1 = os.path.join(ROOT_DIR, 'datasets/class_boxes/60x90/camera_1/7')
save_path = '1.mp4'


def put_box_on_image(image, coordinates, color=(0, 0, 255)):
    bbox = torch.tensor(coordinates, dtype=torch.int)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    color_list = [color]
    image_true = draw_bounding_boxes(
        image, bbox, width=3, colors=color_list, fill=True)
    image = torchvision.transforms.ToPILImage()(image_true)
    return np.array(image, dtype=np.uint8)


vc1 = cv2.VideoCapture()
vc1.open(vid1)
w1 = int(vc1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(vc1.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames1 = int(vc1.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (w1, h1))
box = os.listdir(boxes1)
box = sorted(box)
print(box)
last_frame = 0
for j in range(frames1):
    _, frame1 = vc1.read()
    box1 = os.path.join(boxes1, f"{j}.txt")
    box1 = load_txt(box1)
    diff = frame1 - last_frame
    diff, mask = clean_diff_image(diff, low_color=0, high_color=200)
    if box1 and j:
        box1 = box1[0].split(',')
        box1 = [int(float(c) * w1) if i == 0 or i == 2 else int(float(c) * h1) for i, c in enumerate(box1)]
        frame1 = put_box_on_image(frame1, [box1], (0, 128, 255))
        # diff = frame1 - last_frame

        diff = put_box_on_image(diff, [box1], (0, 128, 255))
        # cut1 = frame1[box1[1]:box1[3], box1[0]:box1[2], :]

        # min_img = np.min(diff, axis=-1)
        # min_img = np.expand_dims(min_img, axis=-1)
        # max_img = np.min(diff, axis=-1)
        # max_img = np.expand_dims(max_img, axis=-1)

        # cut2 = diff[box1[1]:box1[3], box1[0]:box1[2], :]
        # img = np.concatenate([cut1, cut2], axis=0)
    img = np.concatenate([diff, frame1], axis=0)
    print(j, box1, frame1.shape)

    cv2.imshow('1', img)
    cv2.waitKey(100)
    last_frame = frame1
