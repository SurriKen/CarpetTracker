import time
from datetime import datetime
from ultralytics import YOLO

from parameters import *
from utils import logger, time_converter, save_txt
from yolo8 import detect_synchro_video_polygon

model1 = {
    'model_1': YOLO('runs/detect/camera_1_mix+_8n_100ep/weights/best.pt'),
    'model_2': YOLO('runs/detect/camera_2_mix+_8n_100ep/weights/best.pt')
}
model2 = {
    'model_1': YOLO('runs/detect/camera_1_mix++_8n_150ep/weights/best.pt'),
    'model_2': YOLO('runs/detect/camera_2_mix++_8n_150ep/weights/best.pt')
}
models = [
    (model1, "(mix+ 100ep, F% Acc% Sen%)"),
    (model2, "(mix++ 150ep, F% Acc% Sen%)"),
]
video_paths = [
    # {
    #     'model_1': 'videos/sync_test/test 1_cam 1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 1_cam 2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 2_cam 1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 2_cam 2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 3_cam 1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 3_cam 2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 4_cam 1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 4_cam 2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 5_cam 1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 5_cam 2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 6_cam 1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 6_cam 2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 7_cam1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 7_cam2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 8_cam1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 8_cam2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 10_cam1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 10_cam2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 11_cam1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 11_cam2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 12_cam1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 12_cam2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 13_cam1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 13_cam2_sync.mp4',
    # },
    # {
    #     'model_1': 'videos/sync_test/test 14_cam 1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 14_cam 2_sync.mp4',
    #     'save_path': 'temp/test 14.mp4',
    #     'true_count': 157
    # },
    # {
    #     'model_1': 'videos/sync_test/test 15_cam 1_sync.mp4',
    #     'model_2': 'videos/sync_test/test 15_cam 2_sync.mp4',
    #     'save_path': 'temp/test 15.mp4',
    #     'true_count': 143
    # },
    {
        'model_1': 'videos/sync_test/test 16_cam 1_sync.mp4',
        'model_2': 'videos/sync_test/test 16_cam 2_sync.mp4',
        'save_path': 'temp/test 16.mp4',
        'true_count': 168
    },
    {
        'model_1': 'videos/sync_test/test 17_cam 1_sync.mp4',
        'model_2': 'videos/sync_test/test 17_cam 2_sync.mp4',
        'save_path': 'temp/test 17.mp4',
        'true_count': 167
    },
    {
        'model_1': 'videos/sync_test/test 18_cam 1_sync.mp4',
        'model_2': 'videos/sync_test/test 18_cam 2_sync.mp4',
        'save_path': 'temp/test 18.mp4',
        'true_count': 129
    },
    {
        'model_1': 'videos/sync_test/test 19_cam 1_sync.mp4',
        'model_2': 'videos/sync_test/test 19_cam 2_sync.mp4',
        'save_path': 'temp/test 19.mp4',
        'true_count': 136
    },
    {
        'model_1': 'videos/sync_test/test 20_cam 1_sync.mp4',
        'model_2': 'videos/sync_test/test 20_cam 2_sync.mp4',
        'save_path': 'temp/test 20.mp4',
        'true_count': 142
    },
]

for mod in models:
    for i in range(len(video_paths)):
        args = {
            'conf': 0.3, 'iou': 0.,
            'POLY_CAM1_IN': POLY_CAM1_IN, 'POLY_CAM1_OUT': POLY_CAM1_OUT,
            'POLY_CAM2_IN': POLY_CAM2_IN, 'POLY_CAM2_OUT': POLY_CAM2_OUT,
            'start_frame': 0, 'end_frame': 0,
            'MIN_OBJ_SEQUENCE': MIN_OBJ_SEQUENCE, 'MIN_EMPTY_SEQUENCE': MIN_EMPTY_SEQUENCE,
        }
        st = time.time()
        sp = f"{video_paths[i].get('save_path').split('.')[0]} {mod[1]}.mp4" \
            if video_paths[i].get('save_path') else None
        pred_count = detect_synchro_video_polygon(
            models=mod,
            video_paths=video_paths[i],
            save_path=sp,
            start=args['start_frame'],
            finish=args['end_frame'],
            iou=args['iou'],
            conf=args['conf'],
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
        save_txt(txt=msg, txt_path='logs/predict_synch_log.txt', mode='a')

