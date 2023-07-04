import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes

from parameters import ROOT_DIR
from utils import load_txt

FOCUS_CAM1_1 = [[(802, 453), (1526, 562)], [(1116, 202), (1582, 233)]]
FOCUS_CAM1_2 = [[(802, 453), (1116, 202)], [(1526, 562), (1582, 233)]]
# FOCUS_CAM1_2 = [[(410, 840), (758, 515)], [(1635, 216), (1642, 137)]]

FOCUS_CAM2_1 = [[(246, 304), (300, 238)], [(339, 337), (376, 260)]]
FOCUS_CAM2_2 = [[(246, 304), (339, 337)], [(300, 238), (376, 260)]]


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        # raise Exception('lines do not intersect')
        return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


def yolobbox2bbox(x, y, w, h):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return int(x1), int(y1), int(x2), int(y2)


def put_box_on_image(image, coordinates, color=(0, 0, 255)):
    bbox = torch.tensor(coordinates, dtype=torch.int)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    color_list = [color]
    image_true = draw_bounding_boxes(
        image, bbox, width=3, colors=color_list, fill=True)
    image = torchvision.transforms.ToPILImage()(image_true)
    return np.array(image, dtype=np.uint8)


def box_square(box):
    return (box[2] - box[0]) * (box[3] - box[1])


FOC_1 = line_intersection(FOCUS_CAM2_1[0], FOCUS_CAM2_1[1])
FOC_2 = line_intersection(FOCUS_CAM2_2[0], FOCUS_CAM2_2[1])
print(FOC_1, FOC_2)

IMG = cv2.imread(os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_2/init/test 22_02456.png'))
h, w = IMG.shape[:2]
print(IMG.max(), h, w)
# cv2.imshow('image', IMG)
# cv2.waitKey(0)
BOX = load_txt(os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_2/boxes/test 22_02456.txt'))[0].split(' ')
# TARGET = (int(w / 2), int(h / 2))
# TARGET = (810, 355)
TARGET = (271, 118)

BOX = yolobbox2bbox(float(BOX[1]) * w, float(BOX[2]) * h, float(BOX[3]) * w, float(BOX[4]) * h)
print(BOX)
C1 = (int((BOX[2] + BOX[0]) / 2), int((BOX[3] + BOX[1]) / 2))
box_hc = BOX[3] - C1[1]
box_wc = BOX[2] - C1[0]
print('before', BOX, C1, f"sq={box_square(BOX)}, w={BOX[2] - BOX[0]}, h={BOX[3] - BOX[1]}")

C2 = line_intersection(line1=[FOC_1, C1], line2=[FOC_2, TARGET])
box_hc2 = line_intersection(line1=[C2, (C2[0], C2[1] + box_hc)], line2=[FOC_1, (C1[0], C1[1] + box_hc)])
box_wc2 = line_intersection(line1=[C2, (C2[0] + box_wc, C2[1])], line2=[FOC_1, (C1[0] + box_wc, C1[1])])
box_2 = [int(2 * C2[0] - box_wc2[0]), int(2 * C2[1] - box_hc2[1]), int(box_wc2[0]), int(box_hc2[1])]
print('move1', box_2, C2, f"sq={box_square(box_2)}, w={box_2[2] - box_2[0]}, h={box_2[3] - box_2[1]}")

box_hc3 = line_intersection(line1=[TARGET, (TARGET[0], TARGET[1] + box_hc)], line2=[FOC_2, box_hc2])
box_wc3 = line_intersection(line1=[TARGET, (TARGET[0] + box_wc, TARGET[1])], line2=[FOC_2, box_wc2])
box_3 = [int(2 * TARGET[0] - box_wc3[0]), int(2 * TARGET[1] - box_hc3[1]), int(box_wc3[0]), int(box_hc3[1])]
print('move2', box_3, TARGET, f"sq={box_square(box_3)}, w={box_3[2] - box_3[0]}, h={box_3[3] - box_3[1]}")

image = put_box_on_image(IMG, [BOX], color=(0, 0, 255))
image = cv2.circle(image, C1, int(0.01 * h), (255, 0, 0), -1)
image = cv2.circle(image, (C1[0], C1[1] + box_hc), int(0.005 * h), (255, 0, 0), -1)
image = cv2.circle(image, (C1[0] + box_wc, C1[1]), int(0.005 * h), (255, 0, 0), -1)
image = cv2.line(image, FOC_1, TARGET, (255, 200, 100), 1)
image = cv2.line(image, FOC_1, (C1[0], C1[1] + box_hc), (255, 200, 100), 1)
image = cv2.line(image, FOC_1, (C1[0] + box_wc, C1[1]), (255, 200, 100), 1)
image = cv2.line(image, FOC_2, TARGET, (100, 200, 255), 1)
image = cv2.line(image, FOC_2, (C1[0], C1[1] + box_hc), (100, 200, 255), 1)
image = cv2.line(image, FOC_2, (C1[0] + box_wc, C1[1]), (100, 200, 255), 1)

image = put_box_on_image(image, [box_2], color=(0, 128, 255))
image = cv2.circle(image, C2, int(0.01 * h), (0, 255, 0), -1)
image = cv2.circle(image, box_wc2, int(0.005 * h), (0, 255, 0), -1)
image = cv2.circle(image, box_hc2, int(0.005 * h), (0, 255, 0), -1)
image = cv2.line(image, FOC_1, box_wc2, (255, 200, 100), 1)
image = cv2.line(image, FOC_1, box_hc2, (255, 200, 100), 1)
image = cv2.line(image, FOC_2, box_wc2, (100, 200, 255), 1)
image = cv2.line(image, FOC_2, box_hc2, (100, 200, 255), 1)

image = put_box_on_image(image, [box_3], color=(0, 255, 255))
image = cv2.circle(image, TARGET, int(0.01 * h), (255, 0, 127), -1)
image = cv2.circle(image, box_wc3, int(0.005 * h), (255, 0, 127), -1)
image = cv2.circle(image, box_hc3, int(0.005 * h), (255, 0, 127), -1)
image = cv2.line(image, FOC_1, box_wc3, (255, 200, 100), 1)
image = cv2.line(image, FOC_1, box_hc3, (255, 200, 100), 1)
image = cv2.line(image, FOC_2, box_wc3, (100, 200, 255), 1)
image = cv2.line(image, FOC_2, box_hc3, (100, 200, 255), 1)
cv2.imshow('image', image)
cv2.waitKey(0)
