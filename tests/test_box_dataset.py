import os

import numpy as np
from scipy import stats

from parameters import ROOT_DIR
from utils import load_data

NUM_FRAMES = 3


def drop_irrelevant(data: list) -> list:
    z = np.abs(stats.zscore(data))
    out = np.where(z > 1)[0]
    r = list(range(len(data)))
    for idx in out:
        r.pop(r.index(idx))
    data2 = np.array(data)[r].tolist()
    return data2


def get_box_geometry(box: list) -> tuple[list, float, float, float, float]:
    """
    Returns: center, width, height, square, diagonal
    """
    center = [(box[2] + box[0]) / 2, (box[3] + box[1]) / 2]
    width = box[2] - box[0]
    height = box[3] - box[1]
    square = width * height
    diagonal = (width ** 2 + height ** 2) ** 0.5
    return center, width, height, square, diagonal


def get_stat_list(x: list) -> list:
    return [x[0], x[-1], min(x), max(x), np.mean(x), np.std(x)]


def resize_list(sequence, length):
    if len(sequence) >= length:
        idx = list(range(len(sequence)))
        x2 = [idx[0]]
        x2.extend(sorted(np.random.choice(idx[1:-1], size=length - 2, replace=False).tolist()))
        x2.append(idx[-1])
        y = [sequence[i] for i in x2]
    else:
        idx = list(range(len(sequence)))
        add = length - len(idx)
        idx.extend(np.random.choice(idx[1:-1], add))
        idx = sorted(idx)
        y = [sequence[i] for i in idx]
    return y


box_data = load_data(os.path.join(ROOT_DIR, 'tests/class_boxes_14_model3.dict'))
box_stat = {}
# 6 centers, length, mean square, mean w, mean h, mean diagonal
for class_ in box_data.keys():
    box_stat[class_] = {}
    for vid in box_data[class_]['camera_1']:
        box_stat[class_][vid] = {}
        for c in [1, 2]:
            box_stat[class_][vid][c] = {
                'centers': [], 'squares': [], 'w': [], 'h': [], 'diagonal': [], 'len': 0, 'stat square': [],
                'stat w': [], 'stat h': [], 'stat diagonal': [], 'center list': [],
                'square seq': [], 'w seq': [], 'h seq': [], 'diagonal seq': [], 'max square': [],
                'max h': [], 'max w': [],
            }

classes = sorted(list(box_stat.keys()))
filled, empty = {}, {}
for class_ in box_data.keys():
    filled[class_], empty[class_] = 0, 0
    for vid in box_data[class_]['camera_1'].keys():
        for cam in box_data[class_].keys():
            if cam == 'camera_1':
                id = 1
            else:
                id = 2
            track = box_data[class_][cam][vid]
            # print(class_, cam, vid, track)
            if track:
                idx = list(track.keys())
                if len(idx) == 1:
                    track = track[idx[0]]
                else:
                    tr = track[idx[0]]
                    for i in idx[1:]:
                        if len(tr) < len(track[i]):
                            tr = track[i]
                    track = tr
                if len(track) > 2:
                    filled[class_] += 1
                    # print(class_, cam, vid, track)
                    # track = drop_irrelevant(track)
                    # box_stat[class_][vid][id]['len'] = len(track)
                    for box in track:
                        center, width, height, square, diagonal = get_box_geometry(box)
                        # print(class_, cam, vid, round(width, 5), round(height, 5), round(square, 5))
                        box_stat[class_][vid][id]['centers'].append(center)
                        box_stat[class_][vid][id]['squares'].append(square)
                        box_stat[class_][vid][id]['h'].append(height)
                        box_stat[class_][vid][id]['w'].append(width)
                        box_stat[class_][vid][id]['diagonal'].append(diagonal)

                    sq_track = drop_irrelevant(box_stat[class_][vid][id]['squares'])
                    if len(sq_track) > 2:
                        idx = [box_stat[class_][vid][id]['squares'].index(i) for i in sq_track]
                        box_stat[class_][vid][id]['centers'] = np.array(box_stat[class_][vid][id]['centers'])[idx].tolist()
                        box_stat[class_][vid][id]['squares'] = np.array(box_stat[class_][vid][id]['squares'])[idx].tolist()
                        box_stat[class_][vid][id]['h'] = np.array(box_stat[class_][vid][id]['h'])[idx].tolist()
                        box_stat[class_][vid][id]['w'] = np.array(box_stat[class_][vid][id]['w'])[idx].tolist()
                        box_stat[class_][vid][id]['diagonal'] = np.array(box_stat[class_][vid][id]['diagonal'])[
                            idx].tolist()

                        box_stat[class_][vid][id]['center list'] = \
                            resize_list(box_stat[class_][vid][id]['centers'], length=NUM_FRAMES)
                        box_stat[class_][vid][id]['stat square'] = get_stat_list(sq_track)
                        box_stat[class_][vid][id]['stat h'] = get_stat_list(box_stat[class_][vid][id]['h'])
                        box_stat[class_][vid][id]['stat w'] = get_stat_list(box_stat[class_][vid][id]['w'])
                        box_stat[class_][vid][id]['stat diagonal'] = \
                            get_stat_list(box_stat[class_][vid][id]['diagonal'])
                        box_stat[class_][vid][id]['len'] = len(sq_track)
                        # print(class_, cam, vid, box_stat[class_][vid][id])
                        # box_stat[class_][vid][id]['center list'] = \
                        #     resize_list(np.array(box_stat[class_][vid][id]['centers'])[idx].tolist(), length=6)
                        #
                        # print(idx)
                        idx = [box_stat[class_][vid][id]['centers'].index(i) for i in
                               box_stat[class_][vid][id]['center list']]
                        box_stat[class_][vid][id]['square seq'] = np.array(box_stat[class_][vid][id]['squares'])[
                            idx].tolist()
                        box_stat[class_][vid][id]['h seq'] = np.array(box_stat[class_][vid][id]['h'])[idx].tolist()
                        box_stat[class_][vid][id]['w seq'] = np.array(box_stat[class_][vid][id]['w'])[idx].tolist()
                        box_stat[class_][vid][id]['diagonal seq'] = np.array(box_stat[class_][vid][id]['diagonal'])[
                            idx].tolist()
                else:
                    empty[class_] += 1

print('filled', filled)
print('empty', empty)
# data_1 = []
# # 1 - [x1y1 x2y2 x3y3 x4y4 x5y5 x6y6 S1 S2 S3 S4 S5 S6 h1 h2 h3 h4 h5 h6 w1 w2 w3 w4 w5 w6 d1 d2 d3 d4 d5 d6]
# data_2 = []
# # 2 - [x1y1 x2y2 x3y3 x4y4 x5y5 x6y6 S0 S-1 Smin Smax Sav h0 h-1 hmin hmax hav w0 w-1 wmin wmax wav d0 d-1 dmin dmax dmav]
# data_3 = []
# # 3 - [x1y1 x2y2 x3y3 x4y4 x5y5 x6y6 Smax1-6 hmax1-6 wmax1-6]
# for cl in classes:
#     print(cl)
#     cl_id = classes.index(cl)
#     min_len = 10000
#     max_len = 0
#     lens = []
#     for vid in box_stat[cl].keys():
#         arr1, arr2, arr3 = [], [], []
#         rel = [True, True]
#         for c in [1, 2]:
#             temp1, temp2, temp3 = [], [], []
#             if box_stat[cl][vid][c]['center list']:
#                 for bb in box_stat[cl][vid][c]['center list']:
#                     temp1.extend(bb)
#                     temp2.extend(bb)
#                 temp1.extend(box_stat[cl][vid][c]['stat square'][:-1])
#                 temp1.extend(box_stat[cl][vid][c]['stat h'][:-1])
#                 temp1.extend(box_stat[cl][vid][c]['stat w'][:-1])
#                 temp1.extend(box_stat[cl][vid][c]['stat diagonal'][:-1])
#
#                 # temp2.extend(box_stat[cl][vid][c]['center list'])
#                 temp2.extend(box_stat[cl][vid][c]['square seq'])
#                 temp2.extend(box_stat[cl][vid][c]['h seq'])
#                 temp2.extend(box_stat[cl][vid][c]['w seq'])
#                 temp2.extend(box_stat[cl][vid][c]['diagonal seq'])
#
#                 max_sq = sorted(box_stat[cl][vid][c]['squares'], reverse=True)[:NUM_FRAMES]
#                 max_sq = resize_list(max_sq, NUM_FRAMES)
#                 max_id = [box_stat[cl][vid][c]['squares'].index(i) for i in max_sq]
#                 # print('max_sq', cl, vid, c, [round(s, 5) for s in max_sq])
#                 # for bb in max_id:
#                 #     temp3.extend(box_stat[cl][vid][c]['centers'][bb])
#                 temp3.extend(max_sq)
#                 temp3.extend([box_stat[cl][vid][c]['h'][i] for i in max_id])
#                 temp3.extend([box_stat[cl][vid][c]['w'][i] for i in max_id])
#                 # temp3.extend([box_stat[cl][vid][c]['diagonal'][i] for i in max_id])
#
#                 min_len = box_stat[cl][vid][c]['len'] if box_stat[cl][vid][c]['len'] < min_len else min_len
#                 max_len = box_stat[cl][vid][c]['len'] if box_stat[cl][vid][c]['len'] > max_len else max_len
#                 rel[c - 1] = True
#             else:
#                 rel[c - 1] = False
#             if not temp1:
#                 # temp1 = [0] * (len(box_stat[cl][vid][c]['center list']) + len(box_stat[cl][vid][c]['stat square']) +
#                 #                len(box_stat[cl][vid][c]['stat h']) + len(box_stat[cl][vid][c]['stat w']) +
#                 #                len(box_stat[cl][vid][c]['stat diagonal']))
#                 # temp2 = [0] * (len(box_stat[cl][vid][c]['center list']) + len(box_stat[cl][vid][c]['square seq']) +
#                 #                len(box_stat[cl][vid][c]['h seq']) + len(box_stat[cl][vid][c]['w seq']) +
#                 #                len(box_stat[cl][vid][c]['diagonal seq']))
#                 temp1 = [0] * (NUM_FRAMES * 2 + 5 * 4)
#                 temp2 = [0] * NUM_FRAMES * 6
#                 temp3 = [0] * NUM_FRAMES * 3
#             # print(len(temp3), temp3)
#             lens.append(box_stat[cl][vid][c]['len'])
#             arr1.append(np.array(temp1))
#             arr2.append(np.array(temp2))
#             arr3.append(np.array(temp3))
#         # print(len(arr1[0]), len(arr1[1]), len(arr2[0]), len(arr2[1]))
#         if rel != [False, False]:
#             data_1.append((arr1, cl_id))
#             data_2.append((arr2, cl_id))
#             data_3.append((arr3, cl_id))
#     # print(f"min_len={min_len}, max_len={max_len}")
#     # print(f"lens={sorted(lens)}\n")
#
# from random import shuffle
#
# # shuffle(data_1)
# # x, y = list(zip(*data_1))
# # x = np.array(list(x))
# # y = np.array(y)
# # np.save('x_train_stat.npy', x)
# # np.save('y_train_stat.npy', y)
# #
# # shuffle(data_2)
# # x, y = list(zip(*data_2))
# # x = np.array(list(x))
# # y = np.array(y)
# # np.save('x_train_seq.npy', x)
# # np.save('y_train_seq.npy', y)
# #
# shuffle(data_3)
# x, y = list(zip(*data_3))
# x = np.array(list(x))
# y = np.array(y)
# np.save('x_train_max.npy', x, allow_pickle=True)
# np.save('y_train_max.npy', y, allow_pickle=True)
#
# # x = np.load(os.path.join(ROOT_DIR, 'tests/x_train_stat.npy'))
# # y = np.load(os.path.join(ROOT_DIR, 'tests/y_train_stat.npy'))
# print('\nstat', x.shape, y.shape)
#
# from collections import Counter
#
# # xxx = dict(Counter(y.tolist()))
# # print('Total stat', xxx)
# # xxx = dict(Counter(y.tolist()[:int(0.9 * len(y))]))
# # print('Train stat', xxx)
# # xxx = dict(Counter(y.tolist()[int(0.9 * len(y)):]))
# # print('Val stat', xxx)
# #
# # x = np.load(os.path.join(ROOT_DIR, 'tests/x_train_seq.npy'))
# # y = np.load(os.path.join(ROOT_DIR, 'tests/y_train_seq.npy'))
# # print('\nseq', x.shape, y.shape)
# #
# # xxx = dict(Counter(y.tolist()))
# # print('Total seq', xxx)
# # xxx = dict(Counter(y.tolist()[:int(0.9 * len(y))]))
# # print('Train seq', xxx)
# # xxx = dict(Counter(y.tolist()[int(0.9 * len(y)):]))
# # print('Val seq', xxx)
# #
#
# x = np.load(os.path.join(ROOT_DIR, 'tests/x_train_max.npy'), allow_pickle=True)
# y = np.load(os.path.join(ROOT_DIR, 'tests/y_train_max.npy'), allow_pickle=True)
# print('\nmax', x.shape, y.shape)
#
# xxx = dict(Counter(y.tolist()))
# print('Total max', xxx)
# xxx = dict(Counter(y.tolist()[:int(0.9 * len(y))]))
# print('Train max', xxx)
# xxx = dict(Counter(y.tolist()[int(0.9 * len(y)):]))
# print('Val max', xxx)
#
# max_stat = {}
# for i, cl in enumerate(classes):
#     max_stat[i] = {'min': [[], []], 'max': [[], []], 'mean': [[], []]}
#
# for i in range(len(y)):
#     for c in [0, 1]:
#         minz = sorted(x[i][c][:NUM_FRAMES], reverse=True)
#         while minz[-1] == 0:
#             # print(minz)
#             minz.pop(minz.index(0.))
#             if not minz:
#                 break
#         if minz:
#             max_stat[y[i]]['min'][c].append(np.min(minz))
#             max_stat[y[i]]['mean'][c].append(np.mean(minz))
#             max_stat[y[i]]['max'][c].append(np.max(minz))
#
#
# for i, cl in enumerate(classes):
#     print(f'\n{cl}')
#     for c in [0, 1]:
#         # minz = sorted(max_stat[i]['min'][c], reverse=True)
#         # while len(minz) or 0. in minz:
#         #     minz.pop(minz.index(0.))
#         max_stat[i]['min'][c] = [round(s, 4) for s in sorted(max_stat[i]['min'][c])[:5]]
#         max_stat[i]['max'][c] = [round(s, 4) for s in sorted(max_stat[i]['max'][c], reverse=True)[:5]]
#         max_stat[i]['mean'][c] = round(np.mean(max_stat[i]['mean'][c]), 4)
#     # print(f"Min: \n-{max_stat[i]['min'][0]}\n-{max_stat[i]['min'][1]}")
#     # print(f"Max: \n-{max_stat[i]['max'][0]}\n-{max_stat[i]['max'][1]}")
#     print(f"Mean: \n"
#           f"-{max_stat[i]['mean'][0]} range {max_stat[i]['min'][0][0]} - {max_stat[i]['max'][0][0]}\n"
#           f"-{max_stat[i]['mean'][1]} range {max_stat[i]['min'][1][0]} - {max_stat[i]['max'][1][0]}")
