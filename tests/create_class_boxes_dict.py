import os
import time

import numpy as np
from pywt.data import camera
from ultralytics import YOLO
from scipy import stats
from parameters import ROOT_DIR, DATASET_DIR
from utils import save_data, load_data, time_converter
from yolo8 import detect_mono_video_polygon, detect_synchro_video_polygon

# model1 = {
#     'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix+_8n_100ep/weights/best.pt')),
#     'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix+_8n_100ep/weights/best.pt'))
# }
# model2 = {
#     'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix++_8n_150ep/weights/best.pt')),
#     'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix++_8n_150ep/weights/best.pt'))
# }
# model3 = {
#     'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix+++_8n_200ep/weights/best.pt')),
#     'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix+++_8n_200ep/weights/best.pt'))
# }
# model4 = {
#     'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix4+_8n_350ep/weights/best.pt')),
#     'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix4+_8n_350ep/weights/best.pt'))
# }
model5 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix4+_8n_350ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix4+_8n_350ep/weights/best.pt'))
}
# vid = os.path.join(DATASET_DIR, 'datasets/class_videos_10/60x90/camera_2/7.mp4')
# dataset = os.path.join(DATASET_DIR, 'datasets/class_videos_27')
dataset = os.path.join(DATASET_DIR, 'datasets/train_class_videos')
target_model = model5
box_data = {}
# box_data:
#     - '115x200'
#     - '115x400'
#     - '150x300'
#     - '60x90'
#     - '85x150'
#         - video_name
#         ...
#             - camera 1
#             - camera 2
#                 - box sequence
stat = {}
start = time.time()
for class_ in os.listdir(dataset):
    box_data[class_] = {}
    filled, empty, total = 0, 0, 0
    for video in os.listdir(os.path.join(dataset, class_, 'camera_1')):
        box_data[class_][video] = {}
        count, tracks = detect_synchro_video_polygon(
            models=(model5, "(mix4+ 350ep)"),
            video_paths={"model_1": os.path.join(dataset, class_, 'camera_1', video),
                         "model_2": os.path.join(dataset, class_, 'camera_2', video)},
            save_path=os.path.join(ROOT_DIR, 'temp/1.mp4'),
            interactive_video=False,
            save_boxes=False,
            debug=False
        )
        # print('tracks', count, tracks[0]['tr1'])
        if len(tracks) >= 1:
            max_len = 0
            tgid = 0
            for i, v in enumerate(tracks):
                idxs = []
                if v['tr1']:
                    idxs.extend(v['tr1'][0])
                if v['tr2']:
                    idxs.extend(v['tr2'][0])
                idxs = sorted(list(set(idxs)))
                if max_len < idxs[-1] - idxs[0]:
                    max_len = idxs[-1] - idxs[0]
                    tgid = i
            tracks = tracks[tgid]
        else:
            tracks = {'tr1': [], 'tr2': []}
            # for k, v in boxes.items():
            #     boxes_norm = [[[c / shape[i % 2] for i, c in enumerate(coord)] for coord in v[0]], v[1]]
        # print('tracks2', count, tracks['tr2'])
        # box_data[class_][camera][video] = boxes_norm
        box_data[class_][video]['camera_1'] = tracks['tr1']
        box_data[class_][video]['camera_2'] = tracks['tr2']
        # print('box_data[class_][video]', box_data[class_][video])
        if box_data[class_][video] == {c: [] for c in os.listdir(os.path.join(dataset, class_))}:
            empty += 1
        else:
            filled += 1
        total += 1
        min_frame, max_frame = 1000, 0
        boxes_upd = {camera: [] for camera in os.listdir(os.path.join(dataset, class_))}
        for camera in os.listdir(os.path.join(dataset, class_)):
            if box_data[class_][video][camera]:
                if min_frame > min(box_data[class_][video][camera][0]):
                    min_frame = min(box_data[class_][video][camera][0])
                if max_frame < max(box_data[class_][video][camera][0]):
                    max_frame = max(box_data[class_][video][camera][0])
        # print('min_frame, max_frame', min_frame, max_frame)
        for i in range(min_frame, max_frame + 1):
            for camera in os.listdir(os.path.join(dataset, class_)):
                if box_data[class_][video][camera] and i in box_data[class_][video][camera][0]:
                    idx = box_data[class_][video][camera][0].index(i)
                    boxes_upd[camera].append(box_data[class_][video][camera][1][idx])
                else:
                    boxes_upd[camera].append([])
        print(class_, video, "boxes_upd", boxes_upd)
        if boxes_upd != {camera: [] for camera in os.listdir(os.path.join(dataset, class_))}:
            box_data[class_][video] = boxes_upd
        # break
    stat[class_] = dict(total=total, filled=filled, empty=empty)
#     break
# print(box_data)
for k, v in stat.items():
    print(k, v)

save_data(data=box_data, folder_path=os.path.join(ROOT_DIR, 'tests'), filename=f'train_class_boxes_model5_Pex')
print(time_converter(time.time() - start))

# 115x200 {'total': 520, 'filled': 519, 'empty': 1}
# 115x400 {'total': 69, 'filled': 68, 'empty': 1}
# 150x300 {'total': 296, 'filled': 294, 'empty': 2}
# 60x90 {'total': 262, 'filled': 252, 'empty': 10}
# 85x150 {'total': 514, 'filled': 511, 'empty': 3}


