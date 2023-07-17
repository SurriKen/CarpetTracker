import os.path
import time
from datetime import datetime
from ultralytics import YOLO

from nn_classificator import VideoClassifier
from parameters import *
from utils import logger, time_converter, save_txt, get_name_from_link
from yolo8 import detect_synchro_video_polygon

model1 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix+_8n_100ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix+_8n_100ep/weights/best.pt'))
}
model2 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix++_8n_150ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix++_8n_150ep/weights/best.pt'))
}
model3 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix+++_8n_200ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix+++_8n_200ep/weights/best.pt'))
}
model4 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix4+_8n_250ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix4+_8n_250ep/weights/best.pt'))
}
model5 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix4+_8n_350ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix4+_8n_350ep/weights/best.pt'))
}

models = [
    # (model1, "(mix+ 100ep, F% Acc% Sen%)"),
    # (model2, "(mix++ 150ep, F% Acc% Sen%)"),
    # (model3, "(mix+++ 200ep, F% Acc% Sen%)"),
    # (model4, "(mix4+ 250ep)"),
    (model5, "(mix4+ 350ep)"),
]
video_paths = [
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 4_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 4_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 4.mp4',
    #     'true_count': 157
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 5_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 5_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 5.mp4',
    #     'true_count': 170
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 6_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 6_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 6.mp4',
    #     'true_count': 111
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 7_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 7_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 7.mp4',
    #     'true_count': 122
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 8_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 8_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 8.mp4',
    #     'true_count': 247
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 10_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 10_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 10.mp4',
    #     'true_count': 127
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 11_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 11_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 11.mp4',
    #     'true_count': 125
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 14_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 14_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 14.mp4',
    #     'true_count': 157
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 15_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 15_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 15.mp4',
    #     'true_count': 143
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 16_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 16_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 16.mp4',
    #     'true_count': 168
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 17_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 17_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 17.mp4',
    #     'true_count': 167
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 18_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 18_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 18.mp4',
    #     'true_count': 129
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 19_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 19_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 19.mp4',
    #     'true_count': 136
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 20_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 20_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 20.mp4',
    #     'true_count': 142
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 21_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 21_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 21.mp4',
    #     'true_count': 137
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 22_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 22_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 22.mp4',
    #     'true_count': 115
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 23_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 23_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 23.mp4',
    #     'true_count': 130
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 24_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 24_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 24.mp4',
    #     'true_count': 159
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 25_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 25_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 25.mp4',
    #     'true_count': 123
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 26_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 26_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 26.mp4',
    #     'true_count': 132
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 27_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 27_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 27.mp4',
    #     'true_count': 146
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 28_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 28_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 28.mp4',
    #     'true_count': 153
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 29_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 29_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 29.mp4',
    #     'true_count': 130
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 30_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 30_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 30.mp4',
    #     'true_count': 167
    # },
    # {
    #     'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 31_cam 1_sync.mp4'),
    #     'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 31_cam 2_sync.mp4'),
    #     'save_path': 'temp/test 31.mp4',
    #     'true_count': 0
    # },
    {
        'model_1': os.path.join(DATASET_DIR, 'videos/classification_videos/video_sync/test 52_cam 1_sync.mp4'),
        'model_2': os.path.join(DATASET_DIR, 'videos/classification_videos/video_sync/test 52_cam 2_sync.mp4'),
        'save_path': 'temp/test 52.mp4',
        'true_count': 0
    },
]

vc = VideoClassifier(num_classes=5, weights=os.path.join(ROOT_DIR, 'video_class_train/model5_16f_96%/best.pt'))
for mod in models:
    for i in range(len(video_paths)):
        args = {
            'conf': 0.3, 'iou': 0., 'mode': 'standard',
            'POLY_CAM1_IN': POLY_CAM1_IN, 'POLY_CA M1_OUT': POLY_CAM1_OUT,
            'POLY_CAM2_IN': POLY_CAM2_IN, 'POLY_CAM2_OUT': POLY_CAM2_OUT,
            'start_frame': 0, 'end_frame': 0,
            'MIN_OBJ_SEQUENCE': MIN_OBJ_SEQUENCE, 'MIN_EMPTY_SEQUENCE': MIN_EMPTY_SEQUENCE,
        }
        st = time.time()
        spf = ''
        for x in video_paths[i].get('save_path').split('/')[:-1]:
            spf = f"{spf}/{x}"
        sp = f"{spf[1:]}/{get_name_from_link(video_paths[i].get('save_path'))} {mod[1]}.mp4" \
            if video_paths[i].get('save_path') else None
        pred_count = detect_synchro_video_polygon(
            models=mod,
            video_paths=video_paths[i],
            save_path=os.path.join(ROOT_DIR, sp),
            start=args['start_frame'],
            finish=args['end_frame'],
            iou=args['iou'],
            conf=args['conf'],
            interactive_video=True,
            mode=args['mode'],
            save_boxes=True,
            class_model=vc
        )
        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        txt = f"{dt} =========== Predict is finished ===========\n" \
              f"- Model {mod[1]}\n" \
              f"- Video '{video_paths[i]}'\n" \
              f"- True count: '{video_paths[i].get('true_count')}; Predict count: '{pred_count}'\n" \
              f"- Saves as '{sp}'\n" \
              f"- Predict args: {args}\n" \
              f"- Process time: {time_converter(time.time() - st)}\n"
        logger.info(f"Predict is finished. Model {mod[1]}. Video {video_paths[i].get('save_path')}")

        msg = f"{dt}   {txt}\n\n"
        save_txt(txt=msg, txt_path=os.path.join(ROOT_DIR, 'logs/predict_synch_log.txt'), mode='a')
