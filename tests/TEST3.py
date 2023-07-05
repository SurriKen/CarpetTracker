import os

import cv2
import numpy as np
from scipy import stats

from dataset_processing import VideoClass
from parameters import ROOT_DIR
from utils import load_data


def resize_list(sequence, length):
    if len(sequence) >= length:
        idx = list(range(len(sequence)))
        x2 = sorted(np.random.choice(idx, size=length, replace=False).tolist())
        y = [sequence[i] for i in x2]
    else:
        idx = list(range(len(sequence)))
        add = length - len(idx)
        idx.extend(np.random.choice(idx[1:-1], add))
        idx = sorted(idx)
        y = [sequence[i] for i in idx]
    return y

num_frames: int = 6
frame_size: tuple = (128, 128)
concat_axis: int = 2
box_data = load_data(os.path.join(ROOT_DIR, 'tests/class_boxes_10_model3_full.dict'))
box_stat = {}
# 6 centers, length, mean square, mean w, mean h, mean diagonal
vc = VideoClass()
vc.params['split'] = 0.9
vc.params['box_path'] = 'tests/class_boxes_10_model3_full.dict'
dataset = load_data(os.path.join(ROOT_DIR, 'tests/class_boxes_10_model3_full.dict'))
vc.classes = sorted(list(dataset.keys()))

data = []
for class_ in dataset.keys():
    cl_id = vc.classes.index(class_)
    for vid in dataset[class_].keys():
        print(class_, vid)
        seq_frame_1, seq_frame_2 = [], []
        cameras = list(dataset[class_][vid].keys())
        sequence = list(range(len(dataset[class_][vid][cameras[0]])))
        idx = resize_list(sequence, num_frames)
        for fr in range(len(dataset[class_][vid][cameras[0]])):
            fr1 = np.zeros(frame_size)
            fr2 = np.zeros(frame_size)

            if dataset[class_][vid][cameras[0]][fr]:
                box1 = [int(bb * frame_size[i % 2]) for i, bb in enumerate(dataset[class_][vid][cameras[0]][fr])]
                fr1[box1[1]:box1[3], box1[0]:box1[2]] = 1.
                # print(fr1.shape, box1)
            fr1 = np.expand_dims(fr1, axis=-1)
            seq_frame_1.append(fr1)
            # fr1 = np.concatenate([np.expand_dims(fr1, axis=-1), np.expand_dims(fr1, axis=-1), np.expand_dims(fr1, axis=-1)], axis=-1) * 255
            # fr1 = fr1.astype(np.uint8)
            # cv2.imshow('1', fr1)
            # cv2.waitKey(500)

            if dataset[class_][vid][cameras[1]][fr]:
                box2 = [int(bb * frame_size[i % 2]) for i, bb in enumerate(dataset[class_][vid][cameras[1]][fr])]
                fr2[box2[1]:box2[3], box2[0]:box2[2]] = 1.
            fr2 = np.expand_dims(fr2, axis=-1)
            seq_frame_2.append(fr2)
            # fr2 = np.concatenate(
            #     [np.expand_dims(fr2, axis=-1), np.expand_dims(fr2, axis=-1), np.expand_dims(fr2, axis=-1)], axis=-1) * 255
            # fr2 = fr2.astype(np.uint8)
            # cv2.imshow('1', fr1)
            # cv2.waitKey(500)

        seq_frame_1 = np.array(seq_frame_1)[idx]
        seq_frame_2 = np.array(seq_frame_2)[idx]
        if concat_axis is None:
            batch = [[seq_frame_1, seq_frame_2], cl_id]
        elif concat_axis in [0, 1, 2, -1]:
            batch = [np.concatenate([seq_frame_1, seq_frame_2], axis=concat_axis), cl_id]
        else:
            print("Concat_axis is our of range. Choose from None, 0, 1, 2 or -1. Used default value concat_axis=None")
            batch = [[seq_frame_1, seq_frame_2], cl_id]
        # print(seq_frame_1.shape, seq_frame_2.shape, batch[0][0].shape)
        # for i in range(len(seq_frame_1)):
        #     cv2.imshow('1', np.concatenate([batch[0][i].astype(np.uint8), batch[0][i].astype(np.uint8), batch[0][i].astype(np.uint8)], axis=-1))
        #     cv2.waitKey(1000)
        break
    break


