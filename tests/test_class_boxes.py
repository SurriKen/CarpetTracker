import os

from ultralytics import YOLO

from parameters import ROOT_DIR
from yolo8 import detect_mono_video_polygon

model2 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix++_8n_150ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix++_8n_150ep/weights/best.pt'))
}
vid = os.path.join(ROOT_DIR, 'datasets/class_videos_10/60x90/camera_1/7.mp4')
boxes = detect_mono_video_polygon(
    model=model2['model_1'],
    camera=1,
    video_path=vid,
    save_path='',
    interactive_video=False,
    save_boxes_path=None,
    debug=False,
)

for k, v in boxes.items():
    print(k, v)

