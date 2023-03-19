import colorsys
import copy
import io
import os
import pickle
import random
from copy import deepcopy

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

from parameters import MIN_OBJ_SEQUENCE


def save_dict(dict_, file_path, filename):
    with open(os.path.join(file_path, f"{filename}.dict"), 'wb') as handle:
        pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(pickle_path):
    with open(pickle_path, 'rb') as handle:
        b = pickle.load(handle)
    return b


def save_txt(txt, txt_path):
    with open(txt_path, 'w') as f:
        f.write(txt)


def load_txt(txt_path):
    with open(txt_path, 'r') as handle:
        b = handle.readlines()
    return b


def save_yaml(dict_, yaml_path):
    with io.open(yaml_path, 'w', encoding='utf8') as outfile:
        yaml.dump(dict_, outfile, default_flow_style=False, allow_unicode=True)


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded


def get_colors(name_classes: list):
    length = 10 * len(name_classes)
    hsv_tuples = [(x / length, 1., 1.) for x in range(length)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.shuffle(colors)
    return colors[:len(name_classes)]


def get_distance(c1: list, c2: list):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def get_object_count(total_count: list, coords: list):
    res = [[]]
    clust_coords = [[]]
    cnt = 0
    seq_cnt = 0
    for item1, item2 in zip(total_count, total_count[1:]):  # pairwise iteration
        if item2 - item1 < MIN_OBJ_SEQUENCE:
            # The difference is 1, if we're at the beginning of a sequence add both
            # to the result, otherwise just the second one (the first one is already
            # included because of the previous iteration).
            if not res[-1]:  # index -1 means "last element".
                res[-1].extend((item1, item2))
                clust_coords[-1].extend((coords[item1], coords[item2]))
            else:
                res[-1].append(item2)
                clust_coords[-1].append(coords[item2])
                if len(res[-1]) >= MIN_OBJ_SEQUENCE:
                    r = [len(coords[x]) for x in res[-1][-MIN_OBJ_SEQUENCE:]]
                    if seq_cnt < int(np.average(r)):
                        cnt += int(np.average(r)) - seq_cnt
                        seq_cnt = int(np.average(r))
                    if seq_cnt > r[-1] and r[-1] == np.average(r):
                        seq_cnt = r[-1]
        elif res[-1]:
            # The difference isn't 1 so add a new empty list in case it just ended a sequence.
            res.append([])
            clust_coords.append([])
            seq_cnt = 0

    # In case "l" doesn't end with a "sequence" one needs to remove the trailing empty list.
    if not res[-1]:
        del res[-1]
    if not clust_coords[-1]:
        del clust_coords[-1]

    clust_coords_upd = deepcopy(clust_coords)
    for cl in clust_coords_upd:
        if len(cl) < MIN_OBJ_SEQUENCE:
            clust_coords.pop(clust_coords.index(cl))
    return cnt, clust_coords


def add_headline_to_cv_image(image, headline: str):
    if headline:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        font_size = int(im_pil.size[1] * 0.03)
        draw = ImageDraw.Draw(im_pil)
        font = ImageFont.truetype("arial.ttf", font_size)
        label_size = draw.textsize(headline, font)
        text_origin = np.array([int(im_pil.size[0] * 0.01), int(im_pil.size[1] * 0.01)])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=(255, 255, 255)
        )
        draw.text((int(im_pil.size[0] * 0.01), int(im_pil.size[1] * 0.01)), headline, font=font, fill=(0, 0, 0))
        numpy_image = np.array(im_pil)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def get_obj_box_squares(clust_coords):
    bbox = {}
    bb_emp_seq = {}
    for cluster in clust_coords:
        cur_len = 1
        cur_obj = []
        max_idx = 0
        keys = copy.deepcopy(list(bbox.keys()))
        for k in keys:
            if k not in cur_obj and len(bbox[k]) < MIN_OBJ_SEQUENCE:
                bbox.pop(k)
        if bbox:
            max_idx = max(list(bbox.keys()))
        for i, cl in enumerate(cluster):
            if i == 0:
                cur_len = len(cl)
                for k in range(cur_len):
                    bbox[k + max_idx + 1] = [cl[k]]
                    cur_obj.append(k + max_idx + 1)
                    bb_emp_seq[k + max_idx + 1] = 0
            else:
                if cur_len == len(cl) and cur_len == 1:
                    # print('cur_len == len(cl) and cur_len == 1', i, cur_obj)
                    bbox[cur_obj[0]].append(cl[0])
                elif cur_len == len(cl):
                    # print('cur_len == len(cl)', i, cur_obj)
                    box_i = [b for b in range(len(cl))]
                    for k in cur_obj:
                        lb_center = bbox[k][-1][1:3]
                        closest, dist = 0, 1000000
                        for idx in box_i:
                            x = get_distance(lb_center, cl[idx][1:3])
                            if x < dist:
                                dist = x
                                closest = idx
                        box_i.pop(box_i.index(closest))
                        bbox[k].append(cl[closest])
                elif cur_len > len(cl):
                    # print('cur_len > len(cl)', i, cur_obj)
                    box_i = [b for b in range(len(cl))]
                    box_i2 = [b for b in range(len(cl))]
                    cur_obj2 = copy.deepcopy(cur_obj)
                    for b in box_i:
                        lb_center = cl[b][1:3]
                        closest, dist = 0, 1000000
                        for k in cur_obj2:
                            x = get_distance(lb_center, bbox[k][-1][1:3])
                            if x < dist:
                                dist = x
                                closest = k
                        box_i2.pop(box_i2.index(b))
                        cur_obj2.pop(cur_obj2.index(closest))
                        bbox[closest].append(cl[b])
                        if not box_i2:
                            break
                    for k in cur_obj2:
                        cur_obj.pop(cur_obj.index(k))
                        bb_emp_seq[k] += 1
                        if bb_emp_seq[k] == MIN_OBJ_SEQUENCE:
                            cur_obj.pop(cur_obj.index(k))
                    cur_len = len(cl)
                else:
                    # print('cur_len < len(cl)', i, cur_obj)
                    box_i = [b for b in range(len(cl))]
                    for k in cur_obj:
                        if bbox.get(k):
                            lb_center = bbox[k][-1][1:3]
                            closest, dist = 0, 1000000
                            for idx in box_i:
                                x = get_distance(lb_center, cl[idx][1:3])
                                if x < dist:
                                    dist = x
                                    closest = idx
                            box_i.pop(box_i.index(closest))
                            bbox[k].append(cl[closest])
                    for idx in box_i:
                        cur_obj.append(max(cur_obj) + 1)
                        bbox[cur_obj[-1]] = [cl[idx]]
                        bb_emp_seq[cur_obj[-1]] = 0
                    cur_len = len(cl)

    sq = {}
    threshold = 3
    for k in bbox.keys():
        sq[k] = []
        for b in bbox[k]:
            sq[k].append(b[3] * b[4])
        z = np.abs(stats.zscore(sq[k]))
        out = np.where(z > threshold)[0]
        r = list(range(len(sq[k])))
        for idx in out:
            r.pop(r.index(idx))
        sq[k] = np.array(sq[k])[r]

    vecs = []
    for k in sq.keys():
        x = list(zip(sq[k], list(range(len(sq[k])))))
        x = sorted(x, reverse=True)
        x = x[:MIN_OBJ_SEQUENCE]
        x = [i[1] for i in x]
        # print(k, len(np.array(sq[k])[x]), np.mean(np.array(sq[k])[x]))
        vecs.append(np.array(sq[k])[x])
    return np.array(vecs)


if __name__ == '__main__':
    pass
