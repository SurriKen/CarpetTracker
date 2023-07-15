import os
import time

from ultralytics import YOLO

from parameters import *
from utils import get_name_from_link
from yolo8 import detect_synchro_video_polygon

model5 = {
    'model_1': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix4+_8n_350ep/weights/best.pt')),
    'model_2': YOLO(os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix4+_8n_350ep/weights/best.pt'))
}
video_paths = [
    {
        'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 52_cam 1_sync.mp4'),
        'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 52_cam 2_sync.mp4'),
        'save_path': 'temp/test 52.mp4',
        'true_count': 0
    },
]
for i in range(len(video_paths)):
    args = {
        'conf': 0.3, 'iou': 0., 'mode': 'standard',
        'POLY_CAM1_IN': POLY_CAM1_IN, 'POLY_CAM1_OUT': POLY_CAM1_OUT,
        'POLY_CAM2_IN': POLY_CAM2_IN, 'POLY_CAM2_OUT': POLY_CAM2_OUT,
        'start_frame': 0, 'end_frame': 300*25,
        'MIN_OBJ_SEQUENCE': MIN_OBJ_SEQUENCE, 'MIN_EMPTY_SEQUENCE': MIN_EMPTY_SEQUENCE,
    }
    st = time.time()
    spf = ''
    for x in video_paths[i].get('save_path').split('/')[:-1]:
        spf = f"{spf}/{x}"
    sp = f"{spf[1:]}/{get_name_from_link(video_paths[i].get('save_path'))}.mp4" \
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
        save_boxes=True
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