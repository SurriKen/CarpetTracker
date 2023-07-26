import time
from datetime import datetime
from ultralytics import YOLO

from classification.nn_classificator import VideoClassifier
from parameters import *
from utils import time_converter, save_txt, get_name_from_link
from yolo.yolo8 import detect_synchro_video_polygon


def predict(video_paths: dict, stream: bool = False) -> dict:
    vc = VideoClassifier(num_classes=len(CLASSES), weights=CLASSIFICATION_MODEL)
    yolo_models = {
        'model_1': YOLO(YOLO_WEIGTHS.get('model_1')),
        'model_2': YOLO(YOLO_WEIGTHS.get('model_2'))
    }
    args = {
        'conf': 0.3, 'iou': 0., 'POLY_CAM1_IN': POLY_CAM1_IN, 'POLY_CA M1_OUT': POLY_CAM1_OUT,
        'POLY_CAM2_IN': POLY_CAM2_IN, 'POLY_CAM2_OUT': POLY_CAM2_OUT,
        'start_frame': 0, 'end_frame': 0,
        'MIN_OBJ_SEQUENCE': MIN_OBJ_SEQUENCE, 'MIN_EMPTY_SEQUENCE': MIN_EMPTY_SEQUENCE,
    }
    st = time.time()
    spf = ''
    for x in video_paths.get('save_path').split('/')[:-1]:
        spf = f"{spf}/{x}"
    sp = f"{spf[1:]}/{get_name_from_link(video_paths.get('save_path'))}.mp4" \
        if video_paths.get('save_path') else ''
    result = detect_synchro_video_polygon(
        models=yolo_models,
        video_paths=video_paths,
        save_path=sp,
        start=args['start_frame'],
        finish=args['end_frame'],
        interactive_video=True,
        class_model=vc,
        stream=stream,
        debug=True
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
          f"- Saves as '{sp}'\n" \
          f"- Predict args: {args}\n" \
          f"- Process time: {time_converter(time.time() - st)}\n"
    msg = f"{dt}   {txt}\n\n"
    save_txt(txt=msg, txt_path=os.path.join(ROOT_DIR, 'logs/predict_synch_log.txt'), mode='a')
    return result


if __name__ == '__main__':
    video_paths = [
        {
            'model_1': os.path.join(DATASET_DIR, 'videos/sync_test/test 31_cam 1_sync.mp4'),
            'model_2': os.path.join(DATASET_DIR, 'videos/sync_test/test 31_cam 2_sync.mp4'),
            'save_path': os.path.join(ROOT_DIR, 'temp/test 31.mp4'),
        },
    ]
    result = predict(video_paths=video_paths[0])
    print(result)
