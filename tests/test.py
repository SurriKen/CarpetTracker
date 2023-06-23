import copy

import cv2
from parameters import ROOT_DIR
from utils import logger
import os
from skimage import measure
import numpy as np

NONE_ELEMENT_LVL = 0.00001
COLOR_LVL = (0, 255)


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
    # diff = image
    # s = image.shape[0] * image.shape[1]
    # # print(image.shape)
    cleaned_image = np.zeros_like(image)

    # r_img = image[:, :, 0]
    # r_mask = np.where(r_img < COLOR_LVL[1], r_img, 0)
    # r_mask = np.where(r_mask > COLOR_LVL[0], 1, 0)
    # g_img = image[:, :, 1]
    # g_mask = np.where(g_img < COLOR_LVL[1], g_img, 0)
    # g_mask = np.where(g_mask > COLOR_LVL[0], 1, 0)
    # b_img = image[:, :, 2]
    # b_mask = np.where(b_img < COLOR_LVL[1], b_img, 0)
    # b_mask = np.where(b_mask > COLOR_LVL[0], 1, 0)
    # img = r_mask + g_mask + b_mask
    # mask = np.where(img == 3, 1, 0)

    min_img = np.min(image, axis=-1)
    min_img = np.expand_dims(min_img, axis=-1)
    max_img = np.min(image, axis=-1)
    max_img = np.expand_dims(max_img, axis=-1)
    img = np.where(max_img < COLOR_LVL[1], image, (255, 0, 0))
    img = np.where(min_img > COLOR_LVL[0], img, (255, 0, 0))
    gray_mask = np.max(img, axis=-1) - np.min(img, axis=-1)
    mask = np.where(gray_mask < 50, 1, 0)

    # img = np.sum(image, axis=-1) + 1
    # mask = np.where(COLOR_LVL[1] * 3 > img > COLOR_LVL[0] * 3, 1, 0)
    # print(img.max(), np.sum(mask) / s, np.sum(r_mask) / s, np.sum(g_mask) / s, np.sum(b_mask) / s)
    mask = clean_mask_from_noise(mask)
    mask = np.expand_dims(mask, axis=-1)
    cleaned_image = np.where(mask == 0, image, (0, 0, 0))
    # cleaned_image = np.where(color_mask != 0, color_mask, cleaned_image)
    cleaned_image = cleaned_image.astype(np.uint8)
    return cleaned_image


num = 1135
# vid_1 = 'videos/init/test 1_cam 1.mp4'
vid_1 = 'datasets/pylot_dataset/Train_0_0s-300s/video/Train_0.mp4'
vc1 = cv2.VideoCapture()
vc1.open(os.path.join(ROOT_DIR, vid_1))
last_img1 = []
out_size = (640, 360)
start = 0
finish = 7498
out = cv2.VideoWriter(os.path.join(ROOT_DIR, 'temp/test.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 25,
                      (out_size[0], out_size[1] * 2))
for i in range(0, finish):
    _, img1 = vc1.read()
    img1 = cv2.resize(img1, out_size)
    if i >= start and len(last_img1)\
            and np.sum(img1 - last_img1) * 100 / (img1.shape[0] * img1.shape[1] * 255 * 3) < 10:
        if (i + 1) % 100 == 0:
            logger.info(f"Frames {i + 1} / {finish} was processed")
        diff1 = img1 - last_img1
        # cv2.imwrite(f"{ROOT_DIR}/datasets/diff/diff/%05d.png" % i, diff1)
        # image = cv2.imread(f"{ROOT_DIR}/datasets/diff/img/%05d.png" % num)
        # next_image = cv2.imread(f"{ROOT_DIR}/datasets/diff/img/%05d.png" % int(num+1))
        # if np.sum(diff1) * 100 / (img1.shape[0] * img1.shape[1] * 255 * 3) < 10:
        cleaned_image = clean_diff_image(diff1)
        img = np.concatenate((cleaned_image, img1), axis=0)
        # out.write(img)
        cv2.imshow('1', img)
        cv2.waitKey(1)
        last_img1 = img1
    else:
        last_img1 = img1
# out.release()

# vid_1 = 'videos/init/test 1_cam 1.mp4'
# # vid_1 = 'videos/sync_test/test 18_cam 1_sync.mp4'
# out_size = (640, 360)
# out = cv2.VideoWriter(os.path.join(ROOT_DIR, 'temp/test.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 25,
#                       (out_size[0], out_size[1] * 2))
# # out2 = cv2.VideoWriter(os.path.join(ROOT_DIR, 'temp/test2.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 25,
# #                       (out_size[0], out_size[1] * 2))
# vc1 = cv2.VideoCapture()
# vc1.open(os.path.join(ROOT_DIR, vid_1))
# # vc2 = cv2.VideoCapture()
# # vc2.open(os.path.join(ROOT_DIR, vid_2))
# print('fps 1 =', vc1.get(cv2.CAP_PROP_FPS)) #, '\nfps 2 =', vc2.get(cv2.CAP_PROP_FPS))
# start = 0
# finish = int(min([int(vc1.get(cv2.CAP_PROP_FRAME_COUNT)), int(vc1.get(cv2.CAP_PROP_FRAME_COUNT))]) / 10)
# print(start, finish)
# count = 0
# # start, finish = 1100, 1200
# last_img1, last_img2 = [], []
# for i in range(0, finish):
#     _, img1 = vc1.read()
#     # _, img2 = vc2.read()
#     # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     # print(img1.shape)
#     img1 = cv2.resize(img1, out_size)
#     # img1 = np.expand_dims(img1, axis=-1)
#     # img1 = np.concatenate([img1,img1, img1], axis=-1)
#
#     # print(img1.shape)
#     # cv2.imwrite(f"{ROOT_DIR}/datasets/diff/img/%05d.png" % i, img1)
#     # img2 = cv2.resize(img2, out_size)
#
#     if i >= start and len(last_img1) and \
#             np.sum(img1 - last_img1) * 100 / (img1.shape[0] * img1.shape[1] * 255 * 3) < 10:
#         diff1 = img1 - last_img1
#         diff1 = clean_diff_image(diff1)
#         # print(diff1.shape)
#         # cv2.imwrite(f"{ROOT_DIR}/datasets/diff/diff/%05d.png" % i, diff1)
#         # diff2 = img2 - last_img2
#         # print('diff1', diff1.max(), diff1.min(), 'diff2', diff2.max(), diff2.min())
#
#         img = np.concatenate((diff1, img1), axis=0)
#         out.write(img)
#         # img = np.concatenate((img1, img2), axis=0)
#         # out2.write(img)
#         last_img1 = img1
#         # last_img2 = img2
#         if (i + 1) % 100 == 0:
#             logger.info(f"Frames {i + 1} / {finish} was processed")
#     else:
#         last_img1 = img1
#         # last_img2 = img2
# out.release()

# import cv2
#
# image = cv2.imread(f"{ROOT_DIR}/datasets/diff/diff/00420.png")
# print('Image', image.shape)
# cleaned_image = clean_diff_image(image)
# img = np.concatenate((cleaned_image, image), axis=0)
# cv2.imshow('1', img)
# cv2.waitKey(0)