import os
import shutil

from ultralytics import YOLO

from parameters import ROOT_DIR
from utils import load_data

# model1 = {
#     'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix+_8n_100ep/weights/best.pt')),
#     'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix+_8n_100ep/weights/best.pt'))
# }
# class_videos = os.path.join(ROOT_DIR, 'datasets/class_videos')
# class_boxes = os.path.join(ROOT_DIR, 'datasets/class_boxes')
# try:
#     os.mkdir(class_boxes)
# except:
#     shutil.rmtree(class_boxes)
#     os.mkdir(class_boxes)
# cl_list = os.listdir(class_videos)
# camera_1_links, camera_2_links = {cl: [] for cl in cl_list}, {cl: [] for cl in cl_list}
# for cl in cl_list:
#     camera_1_links[cl] = []
#     os.mkdir(os.path.join(class_boxes, cl))
#     os.mkdir(os.path.join(class_boxes, cl, 'camera_1'))
#     os.mkdir(os.path.join(class_boxes, cl, 'camera_2'))
#     c1 = os.listdir(os.path.join(class_videos, cl, 'camera_1'))
#     for vid in c1:
#         camera_1_links[cl].append(os.path.join(class_videos, cl, 'camera_1', vid))
#
#     c2 = os.listdir(os.path.join(class_videos, cl, 'camera_2'))
#     for vid in c2:
#         camera_2_links[cl].append(os.path.join(class_videos, cl, 'camera_2', vid))
#
# for cl in camera_1_links.keys():
#     pass

x = load_data(pickle_path='/media/deny/Новый том/AI/CarpetTracker/tests/boxes/true_bb_1_test 16 (mix++ 150ep, F% Acc% Sen%).dict')
print(x)



