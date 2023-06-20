import os

import cv2
import numpy as np

from parameters import ROOT_DIR
from utils import logger, add_headline_to_cv_image

vid_1 = 'videos/sync_test/test 20_cam 1_sync.mp4'
vid_2 = 'videos/sync_test/test 20_cam 2_sync.mp4'
out_size = (640, 360)
out = cv2.VideoWriter(os.path.join(ROOT_DIR, 'temp/test.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 25,
                      (out_size[0], out_size[1] * 2))
out2 = cv2.VideoWriter(os.path.join(ROOT_DIR, 'temp/test2.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 25,
                      (out_size[0], out_size[1] * 2))
vc1 = cv2.VideoCapture()
vc1.open(os.path.join(ROOT_DIR, vid_1))
vc2 = cv2.VideoCapture()
vc2.open(os.path.join(ROOT_DIR, vid_2))
print('fps 1 =', vc1.get(cv2.CAP_PROP_FPS), '\nfps 2 =', vc2.get(cv2.CAP_PROP_FPS))
# start = 0
# finish = min([int(vc1.get(cv2.CAP_PROP_FRAME_COUNT)), int(vc1.get(cv2.CAP_PROP_FRAME_COUNT))])
# finish = 50
count = 0
start, finish = (0 * 60 + 10) * 25, (0 * 60 + 15) * 25
last_track_seq = {'tr1': [], 'tr2': []}
last_img1, last_img2 = [], []
for i in range(0, finish):
    _, img1 = vc1.read()
    _, img2 = vc2.read()
    img1 = cv2.resize(img1, out_size)
    img2 = cv2.resize(img2, out_size)

    if i >= start and len(last_img1) and len(last_img2):
        diff1 = img1 - last_img1
        diff2 = img2 - last_img2
        # print('diff1', diff1.max(), diff1.min(), 'diff2', diff2.max(), diff2.min())

        img = np.concatenate((diff1, diff2), axis=0)
        out.write(img)
        img = np.concatenate((img1, img2), axis=0)
        out2.write(img)
        last_img1 = img1
        last_img2 = img2
    else:
        last_img1 = img1
        last_img2 = img2
out.release()
