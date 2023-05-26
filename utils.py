import colorsys
import copy
import io
import os
import pickle
import random
from copy import deepcopy
import logging

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

from parameters import MIN_OBJ_SEQUENCE

logging.basicConfig(
    level=logging.DEBUG, filename="py_log.log", filemode="w", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("carpet tracker")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
# logger.debug("carpet tracker started")


def time_converter(time_sec: int) -> str:
    sec = time_sec % 60
    min = time_sec // 60
    hours = min // 60
    min = min % 60
    return f"{hours} h {min} min {sec} sec"

def save_dict(dict_: dict, file_path: str, filename: str) -> None:
    """Save a dictionary to a file"""
    with open(os.path.join(file_path, f"{filename}.dict"), 'wb') as handle:
        pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(pickle_path: str) -> dict:
    """Load a dictionary from saved file"""
    with open(pickle_path, 'rb') as handle:
        b = pickle.load(handle)
    return b


def save_txt(txt: str, txt_path: str) -> None:
    """Save a text to a file"""
    with open(txt_path, 'w') as f:
        f.write(txt)


def load_txt(txt_path: str) -> list[str]:
    """Load a text from saved file"""
    with open(txt_path, 'r') as handle:
        text = handle.readlines()
    return text


def get_colors(name_classes: list) -> list[tuple]:
    """
    Get colors for a given label in list of labels

    Args:
        name_classes: list of labels

    Returns: list of colors as RGB tuples
    """
    length = 10 * len(name_classes)
    hsv_tuples = [(x / length, 1., 1.) for x in range(length)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.shuffle(colors)
    return colors[:len(name_classes)]


def get_distance(c1: list, c2: list) -> float:
    """
    Get distance between two 2D points

    Args:
        c1: (x, y) coordinates of point 1
        c2: (x, y) coordinates of point 2

    Returns: distance

    """
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
        vecs.append(np.array(sq[k])[x])
    return np.array(vecs)


def read_xml(xml_path: str, shrink=False, new_width: int = 416, new_height: int = 416) -> dict:
    with open(xml_path, 'r') as xml:
        lines = xml.readlines()
    xml = ''
    for l in lines:
        xml = f"{xml}{l}"
    filename = xml.split("<filename>")[1].split("</filename>")[0]
    size = xml.split("<size>")[1].split("</size>")[0]
    width = int(size.split("<width>")[1].split("</width>")[0])
    height = int(size.split("<height>")[1].split("</height>")[0])
    objects = xml.split('<object>')[1:]
    coords = []
    for obj in objects:
        name = obj.split("<name>")[1].split("</name>")[0]
        if shrink:
            xmin = int(int(obj.split("<xmin>")[1].split("</xmin>")[0]) / width * new_width)
            ymin = int(int(obj.split("<ymin>")[1].split("</ymin>")[0]) / height * new_height)
            xmax = int(int(obj.split("<xmax>")[1].split("</xmax>")[0]) / width * new_width)
            ymax = int(int(obj.split("<ymax>")[1].split("</ymax>")[0]) / height * new_height)
        else:
            xmin = int(obj.split("<xmin>")[1].split("</xmin>")[0])
            ymin = int(obj.split("<ymin>")[1].split("</ymin>")[0])
            xmax = int(obj.split("<xmax>")[1].split("</xmax>")[0])
            ymax = int(obj.split("<ymax>")[1].split("</ymax>")[0])
        coords.append([xmin, ymin, xmax, ymax, name])
    return {"width": width, "height": height, "coords": coords, "filename": filename}


def remove_empty_xml(xml_folder):
    xml_list = []
    with os.scandir(xml_folder) as fold:
        for f in fold:
            xml_list.append(f.name)
    for xml in xml_list:
        box_info = read_xml(f"{xml_folder}/{xml}")
        if not box_info['coords']:
            os.remove(f"{xml_folder}/{xml}")


if __name__ == '__main__':
    pass
