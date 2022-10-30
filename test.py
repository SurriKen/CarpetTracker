import os
import time

import cv2
import numpy as np
import torch
import torchvision
import PIL
import moviepy
import tensorflow
from PIL import ImageDraw, ImageFont, Image
from torchvision.utils import draw_bounding_boxes

from predict import Predict

VIDEO_PATH = 'videos/Train_0.mp4'

# pred = Predict(
#     video_path=VIDEO_PATH,
#     yolo_model_path='init_frames/Train_0_300s/yolo_model',
#     class_model_path='init_frames/Train_0_300s/class_model',
#     save_path='init_frames/Train_0_300s/predict_Train_full_0.mp4',
#     data_dict_path='init_frames/Train_0_300s/data.dict',
#     yolo_version='v3'
# )

# 30s - processing time 93s
# 60s - processing time 185s

# pred.predict(obj_range=4, headline=True)

# ffmpeg -i input.avi -vcodec libx264 -crf 24 "output.avi"
fpsize = os.path.getsize('init_frames/Train_0_300s/predict_Train_full_0.mp4') / 1024 / 1024
print(fpsize)
os.system('C:/ffmpeg/bin/ffmpeg.exe -i E:/AI/CarpetTracker/init_frames/Train_0_300s/predict_Train_full_0.mp4 -vcodec '
          'libx264 -crf 32 "E:/AI/CarpetTracker/init_frames/Train_0_300s/output.avi"')
# if fpsize >= 150.0:  # Видео размером более 150 КБ необходимо сжать
#
# compress = "C:/ffmpeg/bin/ffmpeg.exe -i {} -r 10 -pix_fmt yuv420p -vcodec libx264 -preset veryslow -profile:v baseline  -crf 23 " \
#            "-acodec aac -b:a 32k -strict -5 {}".format(
#     'init_frames/Train_0_300s/predict_Train_full_0.mp4', 'init_frames/Train_0_300s/predict_Train_full_000.mp4')
# os.system(compress)
