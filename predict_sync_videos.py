import time
from datetime import datetime
from ultralytics import YOLO

from classification.nn_classificator import VideoClassifier
from parameters import *
from utils import time_converter, save_txt
from yolo.yolo8 import detect_synchro_video_polygon


def predict(video_paths: dict, stream: bool = False, save_predict_video: bool = False) -> dict:
    vc = VideoClassifier(num_classes=len(CLASSES), weights=CLASSIFICATION_MODEL)
    yolo_models = {
        'model_1': YOLO(YOLO_WEIGTHS.get('model_1')),
        'model_2': YOLO(YOLO_WEIGTHS.get('model_2'))
    }
    args = {
        'conf': 0.3, 'iou': 0., 'POLY_CAM1_IN': POLY_CAM1_IN, 'POLY_CA M1_OUT': POLY_CAM1_OUT,
        'POLY_CAM2_IN': POLY_CAM2_IN, 'POLY_CAM2_OUT': POLY_CAM2_OUT,
        'start_frame': (0 * 60 + 30) * 0, 'end_frame': (0 * 60 + 31) * 0,
        'MIN_OBJ_SEQUENCE': MIN_OBJ_SEQUENCE, 'MIN_EMPTY_SEQUENCE': MIN_EMPTY_SEQUENCE,
    }
    st = time.time()
    result = detect_synchro_video_polygon(
        models=yolo_models,
        video_paths=video_paths,
        save_path=video_paths.get('save_path'),
        start=args['start_frame'],
        finish=args['end_frame'],
        conf=args['conf'],
        iou=args['iou'],
        interactive_video=True,
        class_model=vc,
        stream=stream,
        debug=False,
        save_predict_video=save_predict_video
    )
    count = result.get('total_count')
    class_dict = result.get('class_distribution')
    class_counter = result.get('class_aequence')
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txt = f"{dt} =========== Predict is finished ===========\n" \
          f"- Video '{video_paths}'\n" \
          f"- Predict count: '{count}'\n" \
          f"- Predict list: '{class_counter}'\n" \
          f"- Predict dict: '{class_dict}'\n" \
          f"- Predict args: {args}\n" \
          f"- Process time: {time_converter(time.time() - st)}\n"
    msg = f"{dt}   {txt}\n\n"
    save_txt(txt=msg, txt_path=os.path.join(ROOT_DIR, 'logs/predict_synch_log.txt'), mode='a')
    return result


if __name__ == '__main__':
    pass
