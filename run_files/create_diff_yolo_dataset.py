import os
import random
import shutil

import numpy as np
from PIL import Image
from skimage import measure

from utils import save_txt, logger

NONE_ELEMENT_LVL = 0.00001
COLOR_LVL = (0, 255)


def clean_mask_from_noise(input_mask):
    labels_mask = measure.label(input_mask)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    sum_area = input_mask.shape[0] * input_mask.shape[1]
    if len(regions) > 1:
        for rg in regions:
            if rg.area / sum_area < NONE_ELEMENT_LVL:
                labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    labels_mask[labels_mask != 0] = 1
    if NONE_ELEMENT_LVL == 1:
        labels_mask = 1 - labels_mask
    else:
        labels_mask = measure.label(1 - labels_mask)
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 1:
            for rg in regions:
                if rg.area / sum_area < NONE_ELEMENT_LVL:
                    labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
        labels_mask[labels_mask != 0] = 1
    return labels_mask


def clean_diff_image(image):
    min_img = np.min(image, axis=-1)
    min_img = np.expand_dims(min_img, axis=-1)
    max_img = np.min(image, axis=-1)
    max_img = np.expand_dims(max_img, axis=-1)
    img = np.where(max_img < COLOR_LVL[1], image, (255, 0, 0))
    img = np.where(min_img > COLOR_LVL[0], img, (255, 0, 0))
    gray_mask = np.max(img, axis=-1) - np.min(img, axis=-1)
    mask = np.where(gray_mask < 50, 1, 0)
    mask = clean_mask_from_noise(mask)
    mask = np.expand_dims(mask, axis=-1)
    cleaned_image = np.where(mask == 0, image, (0, 0, 0))
    cleaned_image = cleaned_image.astype(np.uint8)
    return cleaned_image


def form_dataset_for_train(data: list, split: float, save_path: str, condition=None):
    """
    :param data: list of lists of 2 str and 1 float [[image_folder, corresponding_labels_folder, 0.5], ...]
    :param split: float between 0 and 1
    :param save_path: str
    :param condition: dict
    """
    if condition is None:
        condition = {}
    try:
        os.mkdir(save_path)
        os.mkdir(f"{save_path}/train")
        os.mkdir(f"{save_path}/train/images")
        os.mkdir(f"{save_path}/train/labels")
        os.mkdir(f"{save_path}/val")
        os.mkdir(f"{save_path}/val/images")
        os.mkdir(f"{save_path}/val/labels")
    except:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
        os.mkdir(f"{save_path}/train")
        os.mkdir(f"{save_path}/train/images")
        os.mkdir(f"{save_path}/train/labels")
        os.mkdir(f"{save_path}/val")
        os.mkdir(f"{save_path}/val/images")
        os.mkdir(f"{save_path}/val/labels")

    count = 0
    for folders in data:
        img_list = []
        lbl_list = []

        with os.scandir(folders[0]) as fold:
            for f in fold:
                if f.name[-3:] in ['png', 'jpg']:
                    if condition.get('orig_shape'):
                        img = Image.open(f"{folders[0]}/{f.name}")
                        if img.size == condition.get('orig_shape'):
                            img_list.append(f.name)
                    else:
                        img_list.append(f.name)

        with os.scandir(folders[1]) as fold:
            for f in fold:
                if f.name[-3:] in ['txt']:
                    lbl_list.append(f.name)

        try:
            if 0 < float(folders[2]) <= 1:
                take_num = int(len(img_list) * float(folders[2]))
            else:
                take_num = len(img_list)
        except:
            take_num = len(img_list)

        ids = list(range(len(img_list)))
        z = np.random.choice(ids, take_num, replace=False)
        img_list = [img_list[i] for i in z]
        logger.info(f'\n- img_list: {len(img_list)}\n- lbl_list: {len(lbl_list)}\n')

        random.shuffle(img_list)
        delimiter = int(len(img_list) * split)

        for i, img in enumerate(img_list):
            if i <= delimiter:
                shutil.copy2(f"{folders[0]}/{img}", f"{save_path}/train/images/{count}.jpg")
                if f"{img[:-3]}txt" in lbl_list:
                    shutil.copy2(f"{folders[1]}/{img[:-3]}txt", f"{save_path}/train/labels/{count}.txt")
                else:
                    save_txt(txt='', txt_path=f"{save_path}/train/labels/{count}.txt")
            else:
                shutil.copy2(f"{folders[0]}/{img}", f"{save_path}/val/images/{count}.jpg")
                if f"{img[:-3]}txt" in lbl_list:
                    shutil.copy2(f"{folders[1]}/{img[:-3]}txt", f"{save_path}/val/labels/{count}.txt")
                else:
                    save_txt(txt='', txt_path=f"{save_path}/val/labels/{count}.txt")

            if (count + 1) % 200 == 0:
                logger.info(f"-- prepared {i + 1} images")
            count += 1


data = [
        ['datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01',
         'datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01_', 1.0],
        ['datasets/От разметчиков/batch_02_#110902/obj_train_data/batch_02',
         'datasets/От разметчиков/batch_02_#110902/obj_train_data/batch_02_', 1.0],
        ['datasets/От разметчиков/batch_03_#112497/obj_train_data/batch_03',
         'datasets/От разметчиков/batch_03_#112497/obj_train_data/batch_03_', 1.0],
        ['datasets/От разметчиков/batch_04_#119178/obj_train_data/batch_04',
         'datasets/От разметчиков/batch_04_#119178/obj_train_data/batch_04_', 1.0],
        ['datasets/От разметчиков/batch_05_#147536/batch_05',
         'datasets/От разметчиков/batch_05_#147536/batch_05_', 1.0],
        ['datasets/От разметчиков/batch_mine/obj_train_data/batch_01',
         'datasets/От разметчиков/batch_mine/obj_train_data/batch_01_', 1.0],
]