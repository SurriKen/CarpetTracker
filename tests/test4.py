import os
import shutil
from collections import Counter

import cv2
import numpy as np

from parameters import ROOT_DIR, DATASET_DIR
from utils import load_data

dataset = os.path.join(ROOT_DIR, 'tests/train_class_boxes_model5_Pex.dict')
dataset = load_data(dataset)
video_path = os.path.join(DATASET_DIR, 'datasets')

print(dataset.keys())

cl_stat = {}
for cl in dataset.keys():
    cl_stat[cl] = {cam: [] for cam in dataset[cl][list(dataset[cl].keys())[0]].keys()}
    # print(dataset[cl].keys())
    for vid in dataset[cl].keys():
        for cam in dataset[cl][vid].keys():
            x = []
            for bb in dataset[cl][vid][cam]:
                if bb:
                    x.append(bb)
            cl_stat[cl][cam].append((len(dataset[cl][vid][cam]), vid, cl))
            # cl_stat[cl][cam].append((len(x), vid, cl))
    print(f"\n{cl} data")
    for cam in cl_stat[cl].keys():
        print(cam)

        cl_stat[cl][cam] = sorted(cl_stat[cl][cam])
        lenths, _, _ = list(zip(*cl_stat[cl][cam]))
        # print(f"-- sorted {sorted(cl_stat[cl][cam])}")
        print(f"-- min={min(lenths)}, max={max(lenths)}, av={round(np.mean(lenths), 0)}")
        print(f"-- stat={dict(Counter(lenths))}")
        print(f"-- boxes={dataset[cl][vid][cam]}")
    # break

# print("--------------------------------")
# print(dataset['115x200']['1534.mp4']['camera_1'])
# print(dataset['115x200']['1534.mp4']['camera_2'])


# SAVE_PATH = '/home/deny/Рабочий стол/1'
# DATA_PATH = os.path.join(DATASET_DIR, f'datasets/class_videos_27')
# for k, v in data_dict.items():
#     folder = os.path.join(DATA_PATH, k[0], k[1])
#     for vid in v:
#         print(os.path.join(folder, vid[1]), os.path.join(SAVE_PATH, f"{k[0]}_{k[1]}_{vid[1]}"))
#         shutil.copy(
#             src=os.path.join(folder, vid[1]),
#             dst=os.path.join(SAVE_PATH, f"{k[0]}_{k[1]}_{vid[1]}")
#         )
# folder = '/home/deny/Рабочий стол/1'
# FOLDER_FOR_FRAMES = os.path.join(ROOT_DIR, 'temp/img')
# vid = os.listdir(folder)
# print(vid)
# # from_time - time in video to start cutting, sec (default - 0)
# # to_time - time in video to end cutting, sec (default - 10000), if to_time > frame count -> to_time = frame count
#
# for v in vid:
#     vp = os.path.join(folder, v)
#     video_capture = cv2.VideoCapture()
#     video_capture.open(vp)
#     fps = video_capture.get(cv2.CAP_PROP_FPS)
#     frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
#     print(vp, os.path.isfile(vp), frames)
#     for i in range(int(frames)):
#         ret, frame = video_capture.read()
#         if np.random.random() < 0.5:
#             cv2.imwrite(f"{FOLDER_FOR_FRAMES}/{v.split('.')[0]}_{i}.png", frame)
    # break

# for cl in dataset.keys():
#     # print(dataset[cl].keys())
#     for vid in dataset[cl].keys():
#         for cam in dataset[cl][vid].keys():
#             print(dataset[cl][vid][cam])