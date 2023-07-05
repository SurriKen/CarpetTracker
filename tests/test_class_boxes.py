import os

import numpy as np
from pywt.data import camera
from ultralytics import YOLO
from scipy import stats
from parameters import ROOT_DIR
from utils import save_data, load_data
from yolo8 import detect_mono_video_polygon


model1 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix+_8n_100ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix+_8n_100ep/weights/best.pt'))
}
model2 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix++_8n_150ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix++_8n_150ep/weights/best.pt'))
}
model3 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix+++_8n_200ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix+++_8n_200ep/weights/best.pt'))
}
# vid = os.path.join(ROOT_DIR, 'datasets/class_videos_10/60x90/camera_2/7.mp4')
dataset = os.path.join(ROOT_DIR, 'datasets/class_videos_10')
target_model = model3
box_data = {}
# box_data:
#     - '115x200'
#     - '115x400'
#     - '150x300'
#     - '60x90'
#     - '85x150'
#         - camera 1
#         - camera 2
#             - video_name
#             ...
#                 - dict(track_num: box sequence)
#                 ...
for class_ in os.listdir(dataset):
    box_data[class_] = {}
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
                save_path='',
                interactive_video=False,
                save_boxes_path=None,
                debug=False,
            )
            # print(class_, camera, video, boxes)
            boxes_norm = []
            tgid = 0
            if len(boxes) > 1:
                for k, v in boxes.items():
                    if tgid < len(v[0]):
                        tgid = k
                boxes = {tgid: boxes[tgid]}
            for k, v in boxes.items():
                boxes_norm = [[[c / shape[i % 2] for i, c in enumerate(coord)] for coord in v[0]], v[1]]
            # box_data[class_][camera][video] = boxes_norm
            box_data[class_][video][camera] = boxes_norm
        # print(box_data[class_][video])
        min_frame, max_frame = 1000, 0
        boxes_upd = {camera: [] for camera in os.listdir(os.path.join(dataset, class_))}
        for camera in os.listdir(os.path.join(dataset, class_)):
            if box_data[class_][video][camera]:
            # for camera in os.listdir(os.path.join(dataset, class_)):
                print(box_data[class_][video][camera])
                # if box_data[class_][video][camera][1]:
                if min_frame > min(box_data[class_][video][camera][1]):
                    min_frame = min(box_data[class_][video][camera][1])
                if max_frame < max(box_data[class_][video][camera][1]):
                    max_frame = max(box_data[class_][video][camera][1])
        print(min_frame, max_frame)
        for i in range(min_frame, max_frame+1):
            for camera in os.listdir(os.path.join(dataset, class_)):
                if box_data[class_][video][camera] and i in box_data[class_][video][camera][1]:
                    idx = box_data[class_][video][camera][1].index(i)
                    boxes_upd[camera].append(box_data[class_][video][camera][0][idx])
                else:
                    boxes_upd[camera].append([])
        print(class_, video, "boxes_upd", boxes_upd)
        box_data[class_][video] = boxes_upd


save_data(data=box_data, folder_path=os.path.join(ROOT_DIR, 'tests'), filename=f'class_boxes_10_model3_full')


# def drop_irrelevant(data: list) -> list:
#     z = np.abs(stats.zscore(data))
#     out = np.where(z > 1)[0]
#     r = list(range(len(data)))
#     for idx in out:
#         r.pop(r.index(idx))
#     data2 = np.array(data)[r].tolist()
#     return data2
#
#
# def get_box_geometry(box: list) -> tuple[list, float, float, float, float]:
#     """
#     Returns: center, width, height, square, diagonal
#     """
#     center = [(box[2] + box[0]) / 2, (box[3] + box[1]) / 2]
#     width = box[2] - box[0]
#     height = box[3] - box[1]
#     square = width * height
#     diagonal = (width ** 2 + height ** 2) ** 0.5
#     return center, width, height, square, diagonal
#
#
# def get_stat_list(x: list) -> list:
#     return [x[0], x[-1], min(x), max(x), np.mean(x), np.std(x)]
#
#
# def resize_list(sequence, length):
#     if len(sequence) >= length:
#         idx = list(range(len(sequence)))
#         x2 = [idx[0]]
#         x2.extend(sorted(np.random.choice(idx[1:-1], size=length - 2, replace=False).tolist()))
#         x2.append(idx[-1])
#         y = [sequence[i] for i in x2]
#     else:
#         idx = list(range(len(sequence)))
#         add = length - len(idx)
#         idx.extend(np.random.choice(idx[1:-1], add))
#         idx = sorted(idx)
#         y = [sequence[i] for i in idx]
#     return y
#
#
# box_data = load_data(os.path.join(ROOT_DIR, 'tests/class_boxes_10_model3.dict'))
# box_stat = {}
# # 6 centers, length, mean square, mean w, mean h, mean diagonal
# for class_ in box_data.keys():
#     box_stat[class_] = {}
#     for vid in box_data[class_]['camera_1']:
#         box_stat[class_][vid] = {
#             1: {'centers': [], 'squares': [], 'w': [], 'h': [], 'diagonal': [], 'len': 0, 'stat square': [],
#                 'stat w': [], 'stat h': [], 'stat diagonal': [], 'center list': []},
#             2: {'centers': [], 'squares': [], 'w': [], 'h': [], 'diagonal': [], 'len': 0, 'stat square': [],
#                 'stat w': [], 'stat h': [], 'stat diagonal': [], 'center list': []}
#         }
#
# classes = sorted(list(box_stat.keys()))
# for class_ in box_data.keys():
#     for cam in box_data[class_].keys():
#         if cam == 'camera_1':
#             id = 1
#         else:
#             id = 2
#         for vid in box_data[class_][cam]:
#             track = box_data[class_][cam][vid]
#             # print(class_, cam, vid, track)
#             if track:
#                 idx = list(track.keys())
#                 if len(idx) == 1:
#                     track = track[idx[0]]
#                 else:
#                     tr = track[idx[0]]
#                     for i in idx[1:]:
#                         if len(tr) < len(track[i]):
#                             tr = track[i]
#                     track = tr
#                 if len(track) > 2:
#                     # print(class_, cam, vid, track)
#                     # track = drop_irrelevant(track)
#                     # box_stat[class_][vid][id]['len'] = len(track)
#                     for box in track:
#                         center, width, height, square, diagonal = get_box_geometry(box)
#                         box_stat[class_][vid][id]['centers'].append(center)
#                         box_stat[class_][vid][id]['squares'].append(square)
#                         box_stat[class_][vid][id]['h'].append(height)
#                         box_stat[class_][vid][id]['w'].append(width)
#                         box_stat[class_][vid][id]['diagonal'].append(diagonal)
#
#                     sq_track = drop_irrelevant(box_stat[class_][vid][id]['squares'])
#                     idx = [box_stat[class_][vid][id]['squares'].index(i) for i in sq_track]
#                     box_stat[class_][vid][id]['centers'] = np.array(box_stat[class_][vid][id]['centers'])[idx].tolist()
#                     box_stat[class_][vid][id]['squares'] = np.array(box_stat[class_][vid][id]['squares'])[idx].tolist()
#                     box_stat[class_][vid][id]['h'] = np.array(box_stat[class_][vid][id]['h'])[idx].tolist()
#                     box_stat[class_][vid][id]['w'] = np.array(box_stat[class_][vid][id]['w'])[idx].tolist()
#                     box_stat[class_][vid][id]['diagonal'] = np.array(box_stat[class_][vid][id]['diagonal'])[
#                         idx].tolist()
#
#                     box_stat[class_][vid][id]['center list'] = \
#                         resize_list(box_stat[class_][vid][id]['centers'], length=6)
#                     box_stat[class_][vid][id]['stat square'] = get_stat_list(sq_track)
#                     box_stat[class_][vid][id]['stat h'] = get_stat_list(box_stat[class_][vid][id]['h'])
#                     box_stat[class_][vid][id]['stat w'] = get_stat_list(box_stat[class_][vid][id]['w'])
#                     box_stat[class_][vid][id]['stat diagonal'] = \
#                         get_stat_list(box_stat[class_][vid][id]['diagonal'])
#                     box_stat[class_][vid][id]['len'] = len(sq_track)
                    # print(class_, cam, vid, box_stat[class_][vid][id])
                    # box_stat[class_][vid][id]['center list'] = \
                    #     resize_list(np.array(box_stat[class_][vid][id]['centers'])[idx].tolist(), length=6)
                    #
                    # print(idx)
                    # box_stat['stat square'] = np.array(sq_track)[idx].tolist()
                    # box_stat['stat h'] = np.array(box_stat['h'])[idx].tolist()
                    # box_stat['stat w'] = np.array(box_stat['w'])[idx].tolist()
                    # box_stat['stat diagonal'] = np.array(box_stat['diagonal'])[idx].tolist()
                    # box_stat[class_][vid][id]['center list'] = resize_list(box_stat[class_][vid][id]['centers'], length=6)
                    # box_stat[class_][vid][id]['stat square'] = get_stat_list(box_stat[class_][vid][id]['squares'])
                    # box_stat[class_][vid][id]['stat h'] = get_stat_list(box_stat[class_][vid][id]['h'])
                    # box_stat[class_][vid][id]['stat w'] = get_stat_list(box_stat[class_][vid][id]['w'])
                    # box_stat[class_][vid][id]['stat diagonal'] = get_stat_list(box_stat[class_][vid][id]['diagonal'])

# classes = list(box_stat.keys())
# cam = [1, 2]
# # for cl in classes:
# #     for vid in box_stat[cl].keys():
# #         for c in cam:
# #             print(cl, c, vid, box_stat[cl][vid][c])
# class_stat = {cl: {
#     1: {k: {'start': [], 'end': [], 'min': [], 'max': [], 'mean': []} for k in ['squares', 'h', 'w', 'diagonal']},
#     2: {k: {'start': [], 'end': [], 'min': [], 'max': [], 'mean': []} for k in ['squares', 'h', 'w', 'diagonal']}
# } for cl in classes}
# fill_dict = {cl: dict(fill_both=0, empty_both=0, fill_1_empty_2=0, empty_1_fill_2=0) for cl in classes}
#
# for cl in classes:
#     for vid in box_stat[cl].keys():
#         # print(box_stat[cl][vid].keys())
#         if box_stat[cl][vid][1]['squares'] and box_stat[cl][vid][2]['squares']:
#             fill_dict[cl]['fill_both'] += 1
#         elif box_stat[cl][vid][1]['squares'] and not box_stat[cl][vid][2]['squares']:
#             fill_dict[cl]['fill_1_empty_2'] += 1
#         elif not box_stat[cl][vid][1]['squares'] and box_stat[cl][vid][2]['squares']:
#             fill_dict[cl]['empty_1_fill_2'] += 1
#         else:
#             fill_dict[cl]['empty_both'] += 1
#         for c in cam:
#             # print(cl, c, vid, box_stat[cl][vid][c])
#             for k in ['squares', 'h', 'w', 'diagonal']:
#                 if box_stat[cl][vid][c][k]:
#                     class_stat[cl][c][k]['start'].append(box_stat[cl][vid][c][k][0])
#                     class_stat[cl][c][k]['end'].append(box_stat[cl][vid][c][k][-1])
#                     class_stat[cl][c][k]['min'].append(min(box_stat[cl][vid][c][k]))
#                     class_stat[cl][c][k]['max'].append(max(box_stat[cl][vid][c][k]))
#                     class_stat[cl][c][k]['mean'].append(np.mean(box_stat[cl][vid][c][k]))
# print('\nfill_dict')
# for cl in classes:
#     print(cl, fill_dict[cl])
# print()
#
# for k in ['squares', 'h', 'w', 'diagonal']:
#     for par in ['start', 'end', 'min', 'max', 'mean']:
#         for cl in classes:
#             for c in cam:
#                 if class_stat[cl][c][k]['start']:
#                     # print(class_stat[cl][c][k])
#                     class_stat[cl][c][k][par] = get_stat_list(class_stat[cl][c][k][par])
#                     # class_stat[cl][c][k]['end'] = get_stat_list(class_stat[cl][c][k]['end'])
#                     # class_stat[cl][c][k]['min'] = get_stat_list(class_stat[cl][c][k]['min'])
#                     # class_stat[cl][c][k]['max'] = get_stat_list(class_stat[cl][c][k]['max'])
#                     # class_stat[cl][c][k]['mean'] = get_stat_list(class_stat[cl][c][k]['mean'])
#                 else:
#                     class_stat[cl][c][k][par] = []
#                     # class_stat[cl][c][k]['end'] = []
#                     # class_stat[cl][c][k]['min'] = []
#                     # class_stat[cl][c][k]['max'] = []
#                     # class_stat[cl][c][k]['mean'] = []
#                 print(k, c, cl, par,
#                       f'min={round(class_stat[cl][c][k][par][2], 5)} max={round(class_stat[cl][c][k][par][3], 5)} '
#                       f'mean={round(class_stat[cl][c][k][par][4], 5)}')
#                 # print(k, c, cl, 'end', class_stat[cl][c][k]['end'])
#                 # print(k, c, cl, 'min', class_stat[cl][c][k]['min'])
#                 # print(k, c, cl, 'max', class_stat[cl][c][k]['max'])
#                 # print(k, c, cl, 'mean', class_stat[cl][c][k]['mean'])
#             # print()
#         print()
