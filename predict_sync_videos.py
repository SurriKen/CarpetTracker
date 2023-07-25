import time
from collections import Counter
from datetime import datetime
from ultralytics import YOLO

from classification.nn_classificator import VideoClassifier
from parameters import *
from utils import time_converter, save_txt, get_name_from_link
from yolo.yolo8 import detect_synchro_video_polygon


video_paths = [
    {
        'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 31_cam 1_sync.mp4'),
        'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 31_cam 2_sync.mp4'),
        'save_path': os.path.join(ROOT_DIR, 'temp/test 31.mp4'),
    },
]

vc = VideoClassifier(num_classes=len(CLASSES), weights=CLASSIFICATION_MODEL)
yolo_models = {
    'model_1': YOLO(YOLO_WEIGTHS.get('model_1')),
    'model_2': YOLO(YOLO_WEIGTHS.get('model_2'))
}
for i in range(len(video_paths)):
    args = {
        'conf': 0.3, 'iou': 0., 'POLY_CAM1_IN': POLY_CAM1_IN, 'POLY_CA M1_OUT': POLY_CAM1_OUT,
        'POLY_CAM2_IN': POLY_CAM2_IN, 'POLY_CAM2_OUT': POLY_CAM2_OUT,
        'start_frame': 0, 'end_frame': 0,
        'MIN_OBJ_SEQUENCE': MIN_OBJ_SEQUENCE, 'MIN_EMPTY_SEQUENCE': MIN_EMPTY_SEQUENCE,
    }
    st = time.time()
    spf = ''
    for x in video_paths[i].get('save_path').split('/')[:-1]:
        spf = f"{spf}/{x}"
    sp = f"{spf[1:]}/{get_name_from_link(video_paths[i].get('save_path'))}.mp4" \
        if video_paths[i].get('save_path') else ''
    class_counter = detect_synchro_video_polygon(
        models=yolo_models,
        video_paths=video_paths[i],
        save_path=sp,
        start=args['start_frame'],
        finish=args['end_frame'],
        interactive_video=True,
        class_model=vc,
        debug=False
    )
    count = len(class_counter)
    class_dict = dict(Counter(class_counter))
    dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    txt = f"{dt} =========== Predict is finished ===========\n" \
          f"- Video '{video_paths[i]}'\n" \
          f"- Predict count: '{count}'\n" \
          f"- Predict list: '{class_counter}'\n" \
          f"- Predict dict: '{class_dict}'\n" \
          f"- Saves as '{sp}'\n" \
          f"- Predict args: {args}\n" \
          f"- Process time: {time_converter(time.time() - st)}\n"
    msg = f"{dt}   {txt}\n\n"
    save_txt(txt=msg, txt_path=os.path.join(ROOT_DIR, 'logs/predict_synch_log.txt'), mode='a')
