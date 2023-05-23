import os
from random import shuffle
import time
import numpy as np
import skvideo.io
from collections import Counter

vc_data_path = 'datasets/class_videos'
root_dir = '/Users/denyurchuck/Desktop/CarpetTracker'
split = 0.8


def video_to_array(video_path: str) -> np.ndarray:
    return skvideo.io.vread(os.path.join(root_dir, video_path))


def ohe_from_list(data: list[int], num: int) -> np.ndarray:
    targets = np.array([data]).reshape(-1)
    return np.eye(num)[targets]


def create_video_class_dataset(folder_path: str, split: float) -> dict:
    st = time.time()
    classes = os.listdir(os.path.join(root_dir, folder_path))
    classes = sorted(classes)
    data, lbl, stat_lbl = [], [], []
    stat = dict(train={}, val={}, classes=classes)
    for cl in classes:
        content = os.listdir(os.path.join(root_dir, folder_path, cl))
        content = sorted(content)
        lbl.extend([classes.index(cl)] * len(content))
        for file in content:
            vid = video_to_array(os.path.join(folder_path, cl, file))
            data.append(vid)

    zip_data = list(zip(data, lbl))
    shuffle(zip_data)
    train, val = zip_data[:int(split * len(lbl))], zip_data[int(split * len(lbl)):]
    x_train, y_train = list(zip(*train))
    x_val, y_val = list(zip(*val))

    ytr = dict(Counter(y_train))
    stat_ytr = {}
    for k, v in ytr.items():
        stat_ytr[classes[k]] = v
    stat['train'] = stat_ytr

    yv = dict(Counter(y_val))
    stat_yv = {}
    for k, v in yv.items():
        stat_yv[classes[k]] = v
    stat['val'] = stat_yv

    y_train = ohe_from_list(y_train, len(classes))
    y_val = ohe_from_list(y_val, len(classes))
    print(f"Total dataset processing time = {round(time.time() - st, 1)} sec")
    return dict(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, stat=stat)


data = create_video_class_dataset(vc_data_path, split)
print(data.get('stat'))