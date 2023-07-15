import os
import time

import numpy as np
from pywt.data import camera
from ultralytics import YOLO
from scipy import stats
from parameters import ROOT_DIR, DATASET_DIR
from utils import save_data, load_data, time_converter
from yolo8 import detect_mono_video_polygon

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
model4 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix4+_8n_350ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix4+_8n_350ep/weights/best.pt'))
}
model5 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix4+_8n_350ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix4+_8n_350ep/weights/best.pt'))
}
# vid = os.path.join(DATASET_DIR, 'datasets/class_videos_10/60x90/camera_2/7.mp4')
dataset = os.path.join(DATASET_DIR, 'datasets/class_videos_27')
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
        for camera in os.listdir(os.path.join(dataset, class_)):
            # box_data[class_][camera] = {}
            if camera == 'camera_1':
                model = target_model.get('model_1')
                shape = (1920, 1080)
            else:
                model = target_model.get('model_2')
                shape = (640, 360)
            # for video in os.listdir(os.path.join(dataset, class_, camera)):
            boxes = detect_mono_video_polygon(
                model=model,
                camera=int(camera.split("_")[-1]),
                video_path=os.path.join(dataset, class_, camera, video),
                save_path=os.path.join(ROOT_DIR, f'temp/{camera}/{class_}_{video}'),
                interactive_video=False,
                save_boxes_path=None,
                debug=False,
            )
            # print(class_, camera, video, boxes)
            boxes_norm = []
            if boxes:
                tgid = list(boxes.keys())[0]
                max_len = 0
                if len(boxes) > 1:
                    for k, v in boxes.items():
                        if max_len < len(v[0]):
                            max_len = len(v[0])
                            tgid = k
                    boxes = {tgid: boxes[tgid]}
                for k, v in boxes.items():
                    boxes_norm = [[[c / shape[i % 2] for i, c in enumerate(coord)] for coord in v[0]], v[1]]
            # box_data[class_][camera][video] = boxes_norm
            box_data[class_][video][camera] = boxes_norm
        # print(box_data[class_][video])
        if box_data[class_][video] == {c: [] for c in os.listdir(os.path.join(dataset, class_))}:
            empty += 1
        else:
            filled += 1
        total += 1
        min_frame, max_frame = 1000, 0
        boxes_upd = {camera: [] for camera in os.listdir(os.path.join(dataset, class_))}
        for camera in os.listdir(os.path.join(dataset, class_)):
            if box_data[class_][video][camera]:
                # for camera in os.listdir(os.path.join(dataset, class_)):
                #     print(box_data[class_][video][camera])
                # if box_data[class_][video][camera][1]:
                if min_frame > min(box_data[class_][video][camera][1]):
                    min_frame = min(box_data[class_][video][camera][1])
                if max_frame < max(box_data[class_][video][camera][1]):
                    max_frame = max(box_data[class_][video][camera][1])
        # print(min_frame, max_frame)
        for i in range(min_frame, max_frame + 1):
            for camera in os.listdir(os.path.join(dataset, class_)):
                if box_data[class_][video][camera] and i in box_data[class_][video][camera][1]:
                    idx = box_data[class_][video][camera][1].index(i)
                    boxes_upd[camera].append(box_data[class_][video][camera][0][idx])
                else:
                    boxes_upd[camera].append([])
        print(class_, video, "boxes_upd", boxes_upd)
        if boxes_upd != {camera: [] for camera in os.listdir(os.path.join(dataset, class_))}:
            box_data[class_][video] = boxes_upd
        # break
    stat[class_] = dict(total=total, filled=filled, empty=empty)
    # break

for k, v in stat.items():
    print(k, v)

save_data(data=box_data, folder_path=os.path.join(ROOT_DIR, 'tests'), filename=f'class_boxes_27_model5_Pex')
print(time_converter(time.time() - start))
