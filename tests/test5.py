import os
import time

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from parameters import ROOT_DIR, DATASET_DIR
from utils import save_data, time_converter
from yolo8 import detect_mono_video_polygon


target_model = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix4+_8n_350ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix4+_8n_350ep/weights/best.pt'))
}

# classes = ['115x200', '115x400', '150x300', '60x90', '85x150']
# vid = os.path.join(ROOT_DIR, 'datasets/class_videos_10/60x90/camera_2/7.mp4')
dataset = os.path.join(DATASET_DIR, 'datasets/class_videos_27')
save_path = os.path.join(ROOT_DIR, f'temp/crop_frames_27')
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
if not os.path.isdir(save_path):
    os.mkdir(save_path)
start = time.time()
for class_ in os.listdir(dataset):
    if not os.path.isdir(os.path.join(save_path, f'{class_}')):
        os.mkdir(os.path.join(save_path, f'{class_}'))
    # print(class_)
    box_data[class_] = {}
    filled, empty, total = 0, 0, 0
    for video in os.listdir(os.path.join(dataset, class_, 'camera_1')):
        if not os.path.isdir(os.path.join(save_path, f'{class_}/{video.split(".")[0]}')):
            os.mkdir(os.path.join(save_path, f'{class_}/{video.split(".")[0]}'))
        box_data[class_][video] = {camera: [] for camera in os.listdir(os.path.join(dataset, class_))}
        frames_ = {'camera_1': [], 'camera_2': [], 'id_1': [], 'id_2': []}
        for camera in os.listdir(os.path.join(dataset, class_)):
            save_folder = os.path.join(save_path, f'{class_}/{video.split(".")[0]}/{camera}')
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
            box_data[class_][camera] = {}
            if camera == 'camera_1':
                model = target_model.get('model_1')
                shape = (1920, 1080)
                id = 'id_1'
            else:
                model = target_model.get('model_2')
                shape = (640, 360)
                id = 'id_2'
            # for video in os.listdir(os.path.join(dataset, class_, camera)):
            boxes = detect_mono_video_polygon(
                model=model,
                camera=int(camera.split("_")[-1]),
                video_path=os.path.join(dataset, class_, camera, video),
                save_path=f"",
                interactive_video=False,
                save_boxes_path=None,
                debug=False,
            )
            print(class_, camera, video, boxes)
            boxes_norm = []
            tgid = 0
            max_len = 0
            if len(boxes) > 1:
                for k, v in boxes.items():
                    if max_len < len(v[0]):
                        max_len = len(v[0])
                        tgid = k
                boxes = {tgid: boxes[tgid]}

            vc2 = cv2.VideoCapture()
            vc2.open(os.path.join(dataset, class_, camera, video))
            f = vc2.get(cv2.CAP_PROP_FRAME_COUNT)
            # print("f", f)
            for i in range(int(f)):
                ret, frame = vc2.read()
                # size = (frame.shape[1], frame.shape[0])
                # frame = cv2.resize(frame, (640, 360))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames_[camera].append(frame)
            frames_[id] = [[], []]
            vc2.release()
            cv2.destroyAllWindows()

            for i, frame in enumerate(frames_[camera]):
                for k, v in boxes.items():
                    boxes_norm = [[[c / shape[i % 2] for i, c in enumerate(coord)] for coord in v[0]], v[1]]
                    # frames_[id] = v
                    if i in v[1]:
                        box = v[0][v[1].index(i)]
                        img = frame[box[1]:box[3], box[0]:box[2], :]
                        img = Image.fromarray(img)
                        sp = f"{save_folder}/{i}.png"
                        img.save(sp)
                        # elif np.random.random() < 0.2 and class_ in ['115x200', '150x300', '85x150']:
                        #     img = Image.fromarray(frame)
                        #     sp = f"{ROOT_DIR}/temp/yolo/{class_}_{video.split('.')[0]}_{camera}_{i}.png"
                        #     img.save(sp)
            box_data[class_][video][camera] = boxes_norm
            print(boxes_norm)
    #     break
    # break

        # print(len(frames_['camera_1']), len(frames_['camera_2']), frames_['id_1'], frames_['id_2'])
        if box_data[class_][video] == {c: [] for c in os.listdir(os.path.join(dataset, class_))}:
            empty += 1
        else:
            filled += 1
        total += 1
#         min_frame, max_frame = 1000, 0
#         boxes_upd = {camera: [] for camera in os.listdir(os.path.join(dataset, class_))}
#         for camera in os.listdir(os.path.join(dataset, class_)):
#             # print(box_data[class_][video])
#             if box_data[class_][video][camera]:
#                 # print(box_data[class_][video][camera])
#                 # if box_data[class_][video][camera][1]:
#                 if min_frame > min(box_data[class_][video][camera][1]):
#                     min_frame = min(box_data[class_][video][camera][1])
#                 if max_frame < max(box_data[class_][video][camera][1]):
#                     max_frame = max(box_data[class_][video][camera][1])
#
#         for i in range(min_frame, max_frame+1):
#             for camera in os.listdir(os.path.join(dataset, class_)):
#                 if box_data[class_][video][camera] and i in box_data[class_][video][camera][1]:
#                     idx = box_data[class_][video][camera][1].index(i)
#                     boxes_upd[camera].append(box_data[class_][video][camera][0][idx])
#                 else:
#                     boxes_upd[camera].append([])
#         # print(class_, video, "boxes_upd", boxes_upd)
#         if boxes_upd != {camera: [] for camera in os.listdir(os.path.join(dataset, class_))}:
#             box_data[class_][video] = boxes_upd
#     #     break
#     # break
#     stat[class_] = dict(total=total, filled=filled, empty=empty)
print(time_converter(time.time() - start))
for k, v in stat.items():
    print(k, v)

save_data(data=box_data, folder_path=os.path.join(ROOT_DIR, 'temp'), filename=f'crop_frames_27')

