import os
import random
from collections import Counter

import cv2
import numpy as np

from parameters import ROOT_DIR

# data = os.path.join(ROOT_DIR, 'datasets/class_videos_26')
# classes = ['60x90', '115x400']
#
# cl_files = {}
# av_f = []
# for cl in classes:
#     cl_files[cl] = 0
#     for cam in os.listdir(os.path.join(data, cl)):
#         files = os.listdir(os.path.join(data, cl, cam))
#         for f in files:
#             vc1 = cv2.VideoCapture()
#             vc1.open(os.path.join(data, cl, cam, f))
#             f1 = vc1.get(cv2.CAP_PROP_FRAME_COUNT)
#             cl_files[cl] += f1
#             av_f.append(f1)
#
# print(cl_files)
#
# print(av_f)
# print(f"Len={len(av_f)}, average={np.mean(av_f)}, max={np.max(av_f)}, min={np.min(av_f)}")

import os

import numpy as np

from dataset_processing import VideoClass
from parameters import ROOT_DIR
from utils import save_data, load_data, get_name_from_link


def create_box_video_dataset(
        dataset_path: str, val_split: float, test_split: float, frame_size: tuple = (128, 128)
) -> VideoClass:
    vc = VideoClass()
    vc.params['val_split'] = val_split
    vc.params['test_split'] = test_split
    vc.params['box_path'] = dataset_path
    vc.params['frame_size'] = frame_size
    dataset = load_data(dataset_path)
    vc.dataset = dataset
    vc.params['classes'] = sorted(list(dataset.keys()))
    data = []
    for class_ in dataset.keys():
        cl_id = vc.params['classes'].index(class_)
        for vid in dataset[class_].keys():
            seq_frame_1, seq_frame_2 = [], []
            cameras = sorted(list(dataset[class_][vid].keys()))
            if dataset[class_][vid] != {camera: [] for camera in cameras} and len(
                    dataset[class_][vid][cameras[0]]) > 2:
                sequence = list(range(len(dataset[class_][vid][cameras[0]]))) if len(
                    dataset[class_][vid][cameras[0]]) \
                    else list(range(len(dataset[class_][vid][cameras[1]])))
                for fr in range(len(sequence)):
                    fr1 = np.zeros(frame_size)
                    fr2 = np.zeros(frame_size)

                    if dataset[class_][vid][cameras[0]][fr]:
                        box1 = [int(bb * frame_size[i % 2]) for i, bb in
                                enumerate(dataset[class_][vid][cameras[0]][fr])]
                        fr1[box1[1]:box1[3], box1[0]:box1[2]] = 1.
                    fr1 = np.expand_dims(fr1, axis=-1)
                    seq_frame_1.append(fr1)

                    if dataset[class_][vid][cameras[1]][fr]:
                        box2 = [int(bb * frame_size[i % 2]) for i, bb in
                                enumerate(dataset[class_][vid][cameras[1]][fr])]
                        fr2[box2[1]:box2[3], box2[0]:box2[2]] = 1.
                    fr2 = np.expand_dims(fr2, axis=-1)
                    seq_frame_2.append(fr2)

                seq_frame_1 = np.array(seq_frame_1)
                seq_frame_2 = np.array(seq_frame_2)
                batch = [[seq_frame_1, seq_frame_2], cl_id, (class_, vid)]
                data.append(batch)

    random.shuffle(data)
    x, y, idxs = list(zip(*data))
    # x = np.array(x)
    y = np.array(y)

    train_range = int(1 - (vc.params['val_split'] + vc.params['test_split']) * len(x))
    test_range = int(vc.params['test_split'] * len(x)) if vc.params['test_split'] else 1

    vc.x_train = x[:train_range]
    vc.y_train = y[:train_range]
    vc.params['train_idxs'] = idxs[:train_range]
    vc.params['train_stat'] = dict(Counter(vc.y_train))

    vc.x_val = x[train_range:-test_range]
    vc.y_val = y[train_range:-test_range]
    vc.params['val_idxs'] = idxs[train_range:-test_range]
    vc.params['val_stat'] = dict(Counter(vc.y_val))

    vc.x_test = x[-test_range:]
    vc.y_test = y[-test_range:]
    vc.params['test_idxs'] = idxs[-test_range:]
    vc.params['test_stat'] = dict(Counter(vc.y_val))
    return vc


dataset = create_box_video_dataset(
    dataset_path=os.path.join(ROOT_DIR, 'tests/class_boxes_26_model3_full.dict'),
    val_split=0.1, test_split=0.1, frame_size=(128, 128),
)
print(dataset.x_train[0][0].shape)


def save_dataset(dataset: VideoClass, save_folder: str) -> None:
    name = get_name_from_link(dataset.params['box_path'])
    save_data(dataset.params, save_folder, f"{name}_info")
    save_data(dataset.dataset, save_folder, f"{name}")


def load_dataset(dataset_path: str, dataset_info: str) -> VideoClass:
    vc = VideoClass()
    dataset = load_data(dataset_path)
    vc.params.update(load_data(dataset_info))
    vc.dataset = dataset

    for data_type in ['train', 'val', 'test']:
        data = []
        print(vc.params[f"{data_type}_idxs"])
        for info in vc.params[f"{data_type}_idxs"]:
            class_, vid = info
            seq_frame_1, seq_frame_2 = [], []
            cameras = sorted(list(dataset[class_][vid].keys()))
            sequence = list(range(len(dataset[class_][vid][cameras[0]]))) if len(
                dataset[class_][vid][cameras[0]]) else list(range(len(dataset[class_][vid][cameras[1]])))
            for fr in range(len(sequence)):
                fr1 = np.zeros(vc.params['frame_size'])
                fr2 = np.zeros(vc.params['frame_size'])

                if dataset[class_][vid][cameras[0]][fr]:
                    box1 = [int(bb * vc.params['frame_size'][i % 2]) for i, bb in
                            enumerate(dataset[class_][vid][cameras[0]][fr])]
                    fr1[box1[1]:box1[3], box1[0]:box1[2]] = 1.
                fr1 = np.expand_dims(fr1, axis=-1)
                seq_frame_1.append(fr1)

                if dataset[class_][vid][cameras[1]][fr]:
                    box2 = [int(bb * vc.params['frame_size'][i % 2]) for i, bb in
                            enumerate(dataset[class_][vid][cameras[1]][fr])]
                    fr2[box2[1]:box2[3], box2[0]:box2[2]] = 1.
                fr2 = np.expand_dims(fr2, axis=-1)
                seq_frame_2.append(fr2)
            data.append((np.array([seq_frame_1, seq_frame_2]), vc.params['classes'].index(class_)))
        random.shuffle(data)
        x, y = list(zip(*data))
        setattr(vc, f"x_{data_type}", x)
        setattr(vc, f"y_{data_type}", np.array(y))
    return vc


save_dataset(dataset, save_folder=os.path.join(ROOT_DIR, 'temp'))
# (40, 128, 128, 1)
new = load_dataset(dataset_path=os.path.join(ROOT_DIR, 'temp/class_boxes_26_model3_full.dict'),
                   dataset_info=os.path.join(ROOT_DIR, 'temp/class_boxes_26_model3_full_info.dict'))
# new = load_dataset(new, os.path.join(ROOT_DIR, 'tests'))
print(new.params['classes'])
print(new.x_train[0][0].shape)
