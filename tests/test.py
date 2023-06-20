import cv2
from parameters import ROOT_DIR
from utils import logger
import os
from skimage import measure
import numpy as np

NONE_ELEMENT_LVL = 0.003
COLOR_LVL = 200


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
    diff = image
    # print(image.shape)
    cleaned_image = np.zeros_like(diff)
    r_img = image[:, :, 0]
    r_mask = np.where(r_img > COLOR_LVL, 1, 0)
    g_img = image[:, :, 1]
    g_mask = np.where(g_img > COLOR_LVL, 1, 0)
    b_img = image[:, :, 2]
    b_mask = np.where(b_img > COLOR_LVL, 1, 0)
    # print(np.sum(r_mask), np.sum(g_mask), np.sum(b_mask))
    # img = np.sum(image, axis=-1) + 1
    # img = np.concatenate([np.expand_dims(r_mask, axis=-1), np.expand_dims(g_mask, axis=-1), np.expand_dims(b_mask, axis=-1)], axis=-1)
    img = r_mask + g_mask + b_mask
    # img = np.expand_dims(img, axis=-1)
    mask = np.where(img == 3, 1, 0)
    # print(img.max(), img.shape, np.sum(mask), np.sum(r_mask), np.sum(g_mask), np.sum(b_mask))
    mask = clean_mask_from_noise(mask)
    mask = np.expand_dims(mask, axis=-1)
    color_mask = np.where(mask == 0, (255, 255, 255), (0, 0, 0))
    cleaned_image = np.where(color_mask != 0, color_mask, cleaned_image)
    cleaned_image = cleaned_image.astype(np.uint8)
    return cleaned_image


# num = 1135
vid_1 = 'videos/sync_test/test 18_cam 1_sync.mp4'
vc1 = cv2.VideoCapture()
vc1.open(os.path.join(ROOT_DIR, vid_1))
last_img1 = []
out_size = (480, 240)
start = 200
for i in range(0, 2000):
    _, img1 = vc1.read()
    img1 = cv2.resize(img1, out_size)
    if i >= start and len(last_img1) and \
            np.sum(img1 - last_img1) * 100 / (img1.shape[0] * img1.shape[1] * 255 * 3) < 10:
        diff1 = img1 - last_img1
        next_image = img1
        image = last_img1
        # num = i
        # image = cv2.imread(f"{ROOT_DIR}/datasets/diff/img/%05d.png" % num)
        # next_image = cv2.imread(f"{ROOT_DIR}/datasets/diff/img/%05d.png" % int(num+1))
        if np.sum(next_image - image) * 100 / (image.shape[0] * image.shape[1] * 255 * 3) < 10:
            cleaned_image = clean_diff_image(next_image - image)
            # mask = np.concatenate([mask, mask, mask], axis=-1) * 255
            # mask = mask.astype(np.uint8)
            # print(mask.shape)
            img = np.concatenate((cleaned_image, diff1, image), axis=0)
            # cv2.imwrite('gray.png',out_gray)
            cv2.imshow('1', img)
            cv2.waitKey(1)
    else:
        last_img1 = img1
#         # last_img2 = img2


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
# image = cv2.imread(f"{ROOT_DIR}/datasets/diff/diff/01135.png")
# next_image = cv2.imread(f"{ROOT_DIR}/datasets/diff/diff/01136.png")
# # image2 = cv2.imread(f"{ROOT_DIR}/datasets/diff/diff/00180.png")
# print(np.sum(image), np.sum(next_image))
# print(np.sum(image) * 100 / (image.shape[0] * image.shape[1] * 255 * 3),
#       np.sum(next_image) * 100 / (next_image.shape[0] * next_image.shape[1] * 255 * 3))
# image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
# bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
# out_gray=cv2.divide(image, bg, scale=255)
# out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1]
#
# cv2.imshow('binary', out_binary)
# cv2.imwrite('binary.png',out_binary)
#
# cv2.imshow('gray', out_gray)
# cv2.imwrite('gray.png',out_gray)


#
