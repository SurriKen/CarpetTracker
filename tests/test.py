import copy

import cv2
from parameters import ROOT_DIR
from utils import logger
import os
from skimage import measure
import numpy as np

NONE_ELEMENT_LVL = 0.0001
COLOR_LVL = (0, 245)


def clean_mask_from_noise(input_mask):
    labels_mask = measure.label(input_mask)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    sum_area = input_mask.shape[0] * input_mask.shape[1]
    # print(len(regions), sum_area, input_mask.shape)
    # for rg in regions:
    # if rg.area > 100:
    #     print(rg.area, (min(rg.coords[:, 0]), min(rg.coords[:, 1])),
    #           (max(rg.coords[:, 0]), max(rg.coords[:, 1])), rg.area / sum_area, rg.centroid)
    #     break
    if len(regions) > 1:
        # sum_area = 0
        # for rg in regions:
        #     sum_area += rg.area
        # for rg in regions[1:]:
        for rg in regions:
            if rg.area / sum_area < NONE_ELEMENT_LVL:
                # if rg.area > 1000:
                #     print(rg.area, rg.coords[:, 0], rg.coords[:, 1], rg.area / sum_area, NONE_ELEMENT_LVL)
                labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    labels_mask[labels_mask != 0] = 1
    if NONE_ELEMENT_LVL == 1:
        labels_mask = 1 - labels_mask
    else:
        labels_mask = measure.label(1 - labels_mask)
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        # for rg in regions:
        #     if rg.area > 100:
        #         print(rg.area, (min(rg.coords[:, 0]), min(rg.coords[:, 1])),
        #               (max(rg.coords[:, 0]), max(rg.coords[:, 1])), rg.area / sum_area, rg.centroid)
        if len(regions) > 1:
            # sum_area = 0
            # for rg in regions:
            #     sum_area += rg.area
            for rg in regions:
                # for rg in regions[1:]:
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
    mask = np.where(COLOR_LVL[0] < gray_mask, 1, 0)
    mask = np.where(COLOR_LVL[1] > gray_mask, mask, 0)

    # img = np.sum(image, axis=-1) + 1
    # mask = np.where(COLOR_LVL[1] * 3 > img, 1, 0)
    # print(img.max(), np.sum(mask) / s, np.sum(r_mask) / s, np.sum(g_mask) / s, np.sum(b_mask) / s)
    mask = clean_mask_from_noise(mask)
    mask = np.expand_dims(mask, axis=-1)
    cleaned_image = np.where(mask == 0, image, (0, 0, 0))
    # cleaned_image = np.where(color_mask != 0, color_mask, cleaned_image)
    cleaned_image = cleaned_image.astype(np.uint8)
    return cleaned_image, mask


num = 1135
vid_1 = 'videos/sync_test/test 18_cam 1_sync.mp4'
# vid_1 = 'datasets/pylot_dataset/Train_4_0s-300s/video/Train_4.mp4'
vc1 = cv2.VideoCapture()
vc1.open(os.path.join(ROOT_DIR, vid_1))
last_img1 = []
out_size = (640, 360)
start = 9000
finish = 10000
out = cv2.VideoWriter(os.path.join(ROOT_DIR, 'temp/test.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 25,
                      (out_size[0], out_size[1] * 3))
for i in range(0, finish):
    _, img1 = vc1.read()
    img1 = cv2.resize(img1, out_size)
    if i >= start and len(last_img1)\
            and 15 > np.sum(img1 - last_img1) * 100 / (img1.shape[0] * img1.shape[1] * 255 * 3) > 1:
        if (i + 1) % 10 == 0:
            space = np.sum(img1 - last_img1) * 100 / (img1.shape[0] * img1.shape[1] * 255 * 3)
            logger.info(f"Frames {i + 1} / {finish} was processed. Relevant space {round(space, 2)} %")
        diff1 = img1 - last_img1
        cleaned_image, mask = clean_diff_image(diff1)
        r_img = np.where(mask == 0, img1, (0, 0, 0))
        r_img = r_img.astype(np.uint8)
        # green_hair[(mask == 255).all(-1)] = [0, 255, 0]
        r_img = cv2.addWeighted(r_img, 0.7, img1, 0.3, 0)

        # image = np.where(mask == 0, img1 *0.8 + 0.2 * [255, 0, 0], img1)
        # image = image.astype(np.uint8)
        img = np.concatenate((cleaned_image, r_img, img1), axis=0)
        # out.write(img)
        cv2.imshow('1', img)
        cv2.waitKey(1)
        last_img1 = img1
    else:
        last_img1 = img1
# out.release()

