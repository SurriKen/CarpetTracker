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


def clean_diff_image(image, low_color=0, high_color=255):
    min_img = np.min(image, axis=-1)
    min_img = np.expand_dims(min_img, axis=-1)
    max_img = np.min(image, axis=-1)
    max_img = np.expand_dims(max_img, axis=-1)
    img = np.where(max_img < COLOR_LVL[1], image, (255, 0, 0))
    img = np.where(min_img > COLOR_LVL[0], img, (255, 0, 0))
    gray_mask = np.max(img, axis=-1) - np.min(img, axis=-1)
    mask = np.where(low_color < gray_mask, 1, 0)
    mask = np.where(high_color > gray_mask, mask, 0)

    # img = np.sum(image, axis=-1) + 1
    # mask = np.where(COLOR_LVL[1] * 3 > img, 1, 0)
    # print(img.max(), np.sum(mask) / s, np.sum(r_mask) / s, np.sum(g_mask) / s, np.sum(b_mask) / s)
    mask = clean_mask_from_noise(mask)
    mask = np.expand_dims(mask, axis=-1)
    cleaned_image = np.where(mask == 0, image, (0, 0, 0))
    # cleaned_image = np.where(color_mask != 0, color_mask, cleaned_image)
    cleaned_image = cleaned_image.astype(np.uint8)
    return cleaned_image, mask

vid_links = [
    {
        'model_1': os.path.join(ROOT_DIR, 'videos/sync_test/test 22_cam 1_sync.mp4'),
        'model_2': os.path.join(ROOT_DIR, 'videos/sync_test/test 22_cam 2_sync.mp4'),
    },
    {
        'model_1': os.path.join(ROOT_DIR, 'videos/sync_test/test 23_cam 1_sync.mp4'),
        'model_2': os.path.join(ROOT_DIR, 'videos/sync_test/test 23_cam 2_sync.mp4'),
    },
    {
        'model_1': os.path.join(ROOT_DIR, 'videos/sync_test/test 24_cam 1_sync.mp4'),
        'model_2': os.path.join(ROOT_DIR, 'videos/sync_test/test 24_cam 2_sync.mp4'),
    },
    {
        'model_1': os.path.join(ROOT_DIR, 'videos/sync_test/test 25_cam 1_sync.mp4'),
        'model_2': os.path.join(ROOT_DIR, 'videos/sync_test/test 25_cam 2_sync.mp4'),
    },
    {
        'model_1': os.path.join(ROOT_DIR, 'videos/sync_test/test 26_cam 1_sync.mp4'),
        'model_2': os.path.join(ROOT_DIR, 'videos/sync_test/test 26_cam 2_sync.mp4'),
    },
    {
        'model_1': os.path.join(ROOT_DIR, 'videos/sync_test/test 27_cam 1_sync.mp4'),
        'model_2': os.path.join(ROOT_DIR, 'videos/sync_test/test 27_cam 2_sync.mp4'),
    },
    {
        'model_1': os.path.join(ROOT_DIR, 'videos/sync_test/test 28_cam 1_sync.mp4'),
        'model_2': os.path.join(ROOT_DIR, 'videos/sync_test/test 28_cam 2_sync.mp4'),
    },
    {
        'model_1': os.path.join(ROOT_DIR, 'videos/sync_test/test 29_cam 1_sync.mp4'),
        'model_2': os.path.join(ROOT_DIR, 'videos/sync_test/test 29_cam 2_sync.mp4'),
    },
    {
        'model_1': os.path.join(ROOT_DIR, 'videos/sync_test/test 30_cam 1_sync.mp4'),
        'model_2': os.path.join(ROOT_DIR, 'videos/sync_test/test 30_cam 2_sync.mp4'),
    },
]

x = list(range(200, 225))
a = []
for k in range(1500):
    x = [i + 250 for i in x]
    a.extend(x)

for vid in vid_links:
    for key in vid.keys():
        print(vid.get(key))
        name = vid.get(key).split('/')[-1].split('_')[0]
        if key == 'model_1':
            sp_dif = os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_1/diff')
            sp_mask = os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_1/masked')
            sp_init = os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_1/init')
            sp_cl = os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_1/red')
        else:
            sp_dif = os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_2/diff')
            sp_mask = os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_2/masked')
            sp_init = os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_2/init')
            sp_cl = os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_2/red')
        # vid_1 = 'datasets/pylot_dataset/Train_4_0s-300s/video/Train_4.mp4'
        vc1 = cv2.VideoCapture()
        vc1.open(os.path.join(ROOT_DIR, vid.get(key)))
        f1 = vc1.get(cv2.CAP_PROP_FRAME_COUNT)
        last_img1 = None
        # out_size = (640, 360)
        start = 0
        finish = int(f1)
        # out = cv2.VideoWriter(os.path.join(ROOT_DIR, 'temp/test.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 25,
        #                       (out_size[0], out_size[1] * 3))
        for i in range(0, finish):
            _, img1 = vc1.read()
            # img1 = cv2.resize(img1, out_size)
            if i >= start and i in a and len(last_img1)\
                    and 15 > np.sum(img1 - last_img1) * 100 / (img1.shape[0] * img1.shape[1] * 255 * 3) > 1:
                if (i + 1) % 10 == 0:
                    space = np.sum(img1 - last_img1) * 100 / (img1.shape[0] * img1.shape[1] * 255 * 3)
                    logger.info(f"Frames {i + 1} / {finish} was processed. Relevant space {round(space, 2)} %")
                cv2.imwrite(f"{sp_init}/{name}_%05d.png" % i, img1)
                diff1 = img1 - last_img1
                cleaned_image, mask = clean_diff_image(diff1, high_color=245)
                cv2.imwrite(f"{sp_dif}/{name}_%05d.png" % i, cleaned_image)
                r_img = np.where(mask == 0, (0, 0, 255), (255, 0, 0))
                r_img = r_img.astype(np.uint8)
                # green_hair[(mask == 255).all(-1)] = [0, 255, 0]
                r_img = cv2.addWeighted(r_img, 0.2, img1, 0.8, 0)
                cv2.imwrite(f"{sp_cl}/{name}_%05d.png" % i, r_img)
                b_img = np.where(mask == 0, img1, (0, 0, 0))
                b_img = b_img.astype(np.uint8)
                b_img = cv2.addWeighted(b_img, 0.4, img1, 0.6, 0)
                cv2.imwrite(f"{sp_mask}/{name}_%05d.png" % i, b_img)

                # image = np.where(mask == 0, img1 *0.8 + 0.2 * [255, 0, 0], img1)
                # image = image.astype(np.uint8)
                # img = np.concatenate((cleaned_image, r_img, img1), axis=0)
                # out.write(img)
                # cv2.imshow('1', img)
                # cv2.waitKey(1)

                last_img1 = img1
            else:
                last_img1 = img1
# out.release()

