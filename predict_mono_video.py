import os.path
import time
from datetime import datetime
from ultralytics import YOLO

from parameters import *
from utils import time_converter, save_txt, get_name_from_link
from yolo.yolo8 import detect_mono_video_polygon


model5 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'yolo/camera_1/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'yolo/camera_2/weights/best.pt'))
}

video_paths = [
    {
        'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 30_cam 1_sync.mp4'),
        'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 30_cam 2_sync.mp4'),
        'save_path': os.path.join(ROOT_DIR, 'temp/pred.mp4'),
        'true_count': 0
    },
]

model = 2
for i in range(len(video_paths)):
    args = {
        'conf': 0.3, 'iou': 0., 'mode': 'standard',
        'POLY_CAM1_IN': POLY_CAM1_IN, 'POLY_CA M1_OUT': POLY_CAM1_OUT,
        'POLY_CAM2_IN': POLY_CAM2_IN, 'POLY_CAM2_OUT': POLY_CAM2_OUT,
        'start_frame': 0, 'end_frame': 100,
        'MIN_OBJ_SEQUENCE': MIN_OBJ_SEQUENCE, 'MIN_EMPTY_SEQUENCE': MIN_EMPTY_SEQUENCE,
    }
    st = time.time()
    spf = ''
    for x in video_paths[i].get('save_path').split('/')[:-1]:
        spf = f"{spf}/{x}"
    sp = f"{spf[1:]}/{get_name_from_link(video_paths[i].get('save_path'))}.mp4" \
        if video_paths[i].get('save_path') else None
    pred_count = detect_mono_video_polygon(
        model=model5[f'model_{model}'],
        camera=model,
        video_path=video_paths[i][f'model_{model}'],
        save_path=sp,
        start=args['start_frame'],
        finish=args['end_frame'],
        interactive_video=False,
    )
    dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    txt = f"{dt} =========== Predict is finished ===========\n" \
          f"- Video '{video_paths[i]}'\n" \
          f"- True count: '{video_paths[i].get('true_count')}; Predict count: '{pred_count}'\n" \
          f"- Saves as '{sp}'\n" \
          f"- Predict args: {args}\n" \
          f"- Process time: {time_converter(time.time() - st)}\n"

    msg = f"{dt}   {txt}\n\n"
    save_txt(txt=msg, txt_path=os.path.join(ROOT_DIR, 'logs/predict_synch_log.txt'), mode='a')
