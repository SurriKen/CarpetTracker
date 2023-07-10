import os

import numpy as np

from dataset_processing import VideoClass
from parameters import ROOT_DIR
from tests.test_train_class import VideoClassifier
from utils import save_data, load_data

dataset = VideoClassifier.create_box_video_dataset(
    box_path=os.path.join(ROOT_DIR, 'tests/class_boxes_26_model3_full.dict'),
    split=0.9,
    frame_size=(128, 128),
)

keys = list(dataset.__dict__.keys())
for k in keys:
    x = getattr(dataset, k)
    print(k, type(x), type(x) == np.ndarray)

print(os.path.isfile('tests/x_train.npy'))


def save(dataset: VideoClass, save_path: str):
    keys = list(dataset.__dict__.keys())
    array_keys = ['x_train', 'y_train', 'x_val', 'y_val']
    for k in keys:
        if k not in array_keys and type(getattr(dataset, k)) == np.ndarray:
            array_keys.append(k)
    for k in array_keys:
        arr = np.array(getattr(dataset, k))
        np.save(os.path.join(save_path, f'{k}.npy'), arr, allow_pickle=True)
    dict_ = {}
    for k in keys:
        if k not in array_keys:
            dict_[k] = getattr(dataset, k)
    save_data(dict_, save_path, 'dataset_data')


def load(dataset: VideoClass, folder_path: str) -> VideoClass:
    array_keys = ['x_train', 'y_train', 'x_val', 'y_val']
    for k in array_keys:
        if os.path.isfile(os.path.join(folder_path, f"{k}.npy")):
            arr = np.load(os.path.join(folder_path, f"{k}.npy"), allow_pickle=True)
            setattr(dataset, k, arr)
    if os.path.isfile(os.path.join(folder_path, f"dataset_data.dict")):
        dict_ = load_data(os.path.join(folder_path, f"dataset_data.dict"))
        for k, v in dict_.items():
            setattr(dataset, k, v)
    return dataset


save(dataset, save_path=os.path.join(ROOT_DIR, 'tests'))

new = VideoClass()
new = load(new, os.path.join(ROOT_DIR, 'tests'))
print(new.classes)
print(new.x_train[0][0].shape)
