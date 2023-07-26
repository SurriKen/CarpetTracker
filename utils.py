import colorsys
import copy
import csv
import os
import pickle
import random
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from parameters import MIN_OBJ_SEQUENCE, ROOT_DIR


def time_converter(time_sec: float) -> str:
    if time_sec < 1:
        return f"{round(time_sec * 1000, 1)} ms"
    time_sec = int(time_sec)
    seconds = time_sec % 60
    minutes = time_sec // 60
    hours = minutes // 60
    minutes = minutes % 60
    if hours:
        return f"{hours} h {minutes} min {seconds} sec"
    if minutes:
        return f"{minutes} min {seconds} sec"
    if seconds:
        return f"{seconds} sec"


def plot_and_save_gragh(data: list, xlabel: str, ylabel: str, title: str, save_folder: str) -> None:
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(os.path.join(save_folder, f'{title}.jpg'))
    plt.close()


def save_dict_to_table_txt(data: dict, save_path: str) -> None:
    keys = data.keys()
    file = ''
    n = 0
    for k in keys:
        file = f'{file}{"{:<10} ".format(k)}'
        if len(data.get(k)) > n:
            n = len(data.get(k))
    file = f"{file}\n"

    for i in range(n):
        for k in data.keys():
            file = f'{file}{"{:<10} ".format(data.get(k)[i])}'
        file = f"{file}\n"
    save_txt(txt=file[:-2], txt_path=save_path)


def save_data(data, folder_path: str, filename: str) -> None:
    """Save a dictionary to a file"""
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    with open(os.path.join(folder_path, f"{filename}.dict"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def dict_to_csv(data: dict, folder_path: str, filename: str) -> None:
    """Save a dictionary to a file"""
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    with open(f'{folder_path}/{filename}.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(data.keys())
        w.writerow(data.values())


def load_data(pickle_path: str):
    """Load a dictionary from saved file"""
    with open(pickle_path, 'rb') as handle:
        b = pickle.load(handle)
    return b


def save_txt(txt: str, txt_path: str, mode: str = 'w') -> None:
    """Save a text to a file"""
    with open(txt_path, mode) as f:
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


def add_headline_to_cv_image(image, headline: str):
    if headline:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        font_size = int(im_pil.size[0] * 0.02)
        draw = ImageDraw.Draw(im_pil)
        font = ImageFont.truetype(os.path.join(ROOT_DIR, "arial.ttf"), font_size)
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


def get_name_from_link(link: str):
    vn = link.split('/')[-1].split('.')[:-1]
    video_name = ''
    for v in vn:
        video_name = f"{video_name}.{v}"
    return video_name[1:]
