import os
import pickle
import shutil
import time
from datetime import datetime

import cv2
import numpy as np
import wget
from ultralytics import YOLO

from dataset_processing import DatasetProcessing
from tracker import Tracker
from parameters import SPEED_LIMIT_PERCENT, IMAGE_IRRELEVANT_SPACE_PERCENT, MIN_OBJ_SEQUENCE, MIN_EMPTY_SEQUENCE
from utils import get_colors, load_data, add_headline_to_cv_image, logger, time_converter, save_txt, save_data

yolov8_types = {
    "yolov8n": {"Test Size": 640, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"},
    "yolov8s": {"Test Size": 640, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"},
    "yolov8m": {"Test Size": 1280, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"},
    "yolov8I": {"Test Size": 1280, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"},
    "yolov8x": {"Test Size": 1280, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"},
}


def load_yolo_v8(v8_mode="yolov8n") -> None:
    """

    Args:
        v8_mode:

    Returns:

    """

    if not os.path.isdir(f"yolov8/{v8_mode}.pt"):
        url = yolov8_types[v8_mode]["link"]
        wget.download(url, f"yolov8/{v8_mode}.pt")


def load_kmeans_model(path, name, dict_=False):
    with open(f"{path}/{name}.pkl", "rb") as f:
        model = pickle.load(f)
    lbl_dict = {}
    if dict_:
        lbl_dict = load_data(pickle_path=f"{path}/{name}.dict")
    return model, lbl_dict


def detect_video(model, video_path, save_path, remove_perimeter_boxes=False):
    # Kmeans_model, Kmeans_cluster_names = load_kmeans_model(
    #     path=KMEANS_MODEL_FOLDER,
    #     dict_=True,
    #     name=KMEANS_MODEL_NAME,
    # )

    # Get names and colors
    names = ['carpet']
    colors = get_colors(names)

    # video_path = '/home/deny/Рабочий стол/CarpetTracker/videos/Test_0.mp4'
    # save_path = 'tracked_video.mp4'
    tracker = Tracker()

    vc = cv2.VideoCapture()
    vc.open(video_path)
    f = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Reformat video on 25 fps
    if fps != 25:
        shutil.move(video_path, 'test.mp4')
        DatasetProcessing.change_fps(
            video_path='test.mp4',
            save_path=video_path,
            set_fps=25
        )
        os.remove('test.mp4')
        vc = cv2.VideoCapture()
        vc.open(video_path)
        f = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vc.get(cv2.CAP_PROP_FPS)

    # model = YOLO('runs/detect/train21/weights/best.pt')
    if save_path:
        print(save_path, fps, (w, h))
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))

    result = {}
    for i in range(int(f)):
        print(f"Processed {i + 1} / {f} frames")
        ret, frame = vc.read()
        res = model(frame)
        result[i] = {'boxes': res[0].boxes, 'orig_shape': res[0].orig_shape}
        tracker.process(predict=res, remove_perimeter_boxes=remove_perimeter_boxes)

        if save_path:
            if tracker.id_coords[-1]:
                tracker_id = tracker.id_coords[-1]
                # if tracker_id < len(res[0].boxes):
                #     for box in res[0].boxes:
                #         if

                coords = tracker.coordinates[-1]
                # print(tracker_id, res[0].boxes)
                labels = [
                    f"# {tracker_id[i]} {model.model.names[coords[i][-1]]} {coords[i][-2]:0.2f}"
                    for i in range(len(tracker.coordinates[-1]))
                ]

                if len(labels) > 1:
                    cl = colors * len(labels)
                else:
                    cl = colors

                fr = tracker.put_box_on_image(
                    save_path=None,
                    results=res,
                    labels=labels,
                    color_list=cl,
                    coordinates=tracker.coordinates
                )
                fr = np.array(fr)
            else:
                fr = res[0].orig_img[:, :, ::-1].copy()

            headline = f"Обнаружено объектов: {tracker.obj_count}\n"
            fr = add_headline_to_cv_image(
                image=fr,
                headline=headline
            )
            cv_img = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            out.write(cv_img)
            # if i > 10:
            #     break


def detect_synchro_video(
        models: dict,
        video_paths: dict,
        save_path: str,
        remove_perimeter_boxes=None,
        start: int = 0,
        finish: int = 0,
        speed_limit: float = SPEED_LIMIT_PERCENT,
        iou: float = 0.3,
        conf: float = 0.5
):
    """
    Detect two synchronized videos and save them as one video with boxes to save_path.

    Args:
        models: {'model_1': model_1, "model_2": model_2}
        video_paths: {'model_1': path_1, "model_2": path_2}
        save_path: save_path
        remove_perimeter_boxes: {'model_1': True, "model_2": False}

    Returns:

    """
    # Get names and colors
    if remove_perimeter_boxes is None:
        remove_perimeter_boxes = {'model_1': True, "model_2": False}
    names = ['carpet']
    colors = get_colors(names)

    tracker_1 = Tracker()
    tracker_2 = Tracker()

    vc1 = cv2.VideoCapture()
    vc1.open(video_paths.get("model_1"))
    f1 = vc1.get(cv2.CAP_PROP_FRAME_COUNT)
    fps1 = vc1.get(cv2.CAP_PROP_FPS)
    w1 = int(vc1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(vc1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vc2 = cv2.VideoCapture()
    vc2.open(video_paths.get("model_2"))
    f2 = vc2.get(cv2.CAP_PROP_FRAME_COUNT)
    fps2 = vc2.get(cv2.CAP_PROP_FPS)
    w2 = int(vc2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(vc2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    w = min([w1, w2])
    h = min([h1, h2])

    # Reformat video on 25 fps
    if fps1 != 25:
        shutil.move(video_paths.get("model_1"), 'test.mp4')
        DatasetProcessing.change_fps(
            video_path='test.mp4',
            save_path=video_paths.get("model_1"),
            set_fps=25
        )
        os.remove('test.mp4')
        vc1 = cv2.VideoCapture()
        vc1.open(video_paths.get("model_1"))
        f1 = vc1.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps2 != 25:
        shutil.move(video_paths.get("model_2"), 'test.mp4')
        DatasetProcessing.change_fps(
            video_path='test.mp4',
            save_path=video_paths.get("model_2"),
            set_fps=25
        )
        os.remove('test.mp4')
        vc2 = cv2.VideoCapture()
        vc2.open(video_paths.get("model_2"))
        f2 = vc2.get(cv2.CAP_PROP_FRAME_COUNT)

    # model = YOLO('runs/detect/train21/weights/best.pt')
    if save_path:
        # print(save_path, fps, (w, h))
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (w, h * 2))

    f = min([f1, f2])
    finish = int(f) if finish == 0 or finish < start else finish
    cur_bb_1, cur_bb_2 = [], []
    true_bb_1, true_bb_2 = [], []
    cur_count = 0
    patterns = []
    for i in range(0, finish):
        fr_time = time.time()
        logger.info(f'-- Process {i + 1} / {finish} frame')
        ret1, frame1 = vc1.read()
        ret2, frame2 = vc2.read()

        if i >= start:
            logger.info(f"Processed {i + 1} / {f} frames")
            res1 = models.get('model_1').predict(frame1, iou=iou, conf=conf)
            result = {'boxes': res1[0].boxes.data.tolist(), 'orig_shape': res1[0].orig_shape}
            true_bb_1.append(res1[0].boxes.data.tolist())
            tracker_1.process(
                frame_id=i,
                predict=result,
                remove_perimeter_boxes=remove_perimeter_boxes.get('model_1'),
                speed_limit_percent=speed_limit
            )

            res2 = models.get('model_2').predict(frame2, iou=iou, conf=conf)
            result = {'boxes': res2[0].boxes.data.tolist(), 'orig_shape': res2[0].orig_shape}
            true_bb_2.append(res2[0].boxes.data.tolist())
            tracker_2.process(
                frame_id=i,
                predict=result,
                remove_perimeter_boxes=remove_perimeter_boxes.get('model_2'),
                speed_limit_percent=speed_limit
            )

            input = Tracker.join_frame_id(tracker_dict_1=tracker_1.tracker_dict, tracker_dict_2=tracker_2.tracker_dict)
            cur_bb_1.append(tracker_1.current_boxes)
            cur_bb_2.append(tracker_2.current_boxes)
            patterns = Tracker.get_pattern(input=input)

            # patterns = Tracker.update_pattern(
            #     pattern=patterns,
            #     tracker_1_dict=tracker_1.tracker_dict,
            #     tracker_2_dict=tracker_2.tracker_dict
            # )
            # print("time Tracker.update_pattern:", time_converter(time.time() - x))
            # if patterns:
            #     # print("test_patterns:", test_patterns)
            #     old_pattern_count, patterns, old_pat = Tracker.clean_tracks(
            #         frame=i,
            #         pattern=patterns,
            #         tracker_1_dict=tracker_1.tracker_dict,
            #         tracker_2_dict=tracker_2.tracker_dict
            #     )
            #     cur_count += old_pattern_count
            # if old_pat:
            #     old_patterns.extend(old_pat)
            # x = time.time()
            if save_path:
                if tracker_1.current_id:
                    tracker_id = tracker_1.current_id
                    coords = tracker_1.current_boxes
                    labels = [
                        f"# {tracker_id[tr]} {models.get('model_1').model.names[coords[tr][-1]]} {coords[tr][-2]:0.2f}"
                        for tr in range(len(tracker_id))
                    ]

                    if len(labels) > 1:
                        cl = colors * len(labels)
                    else:
                        cl = colors

                    fr1 = tracker_1.put_box_on_image(
                        save_path=None,
                        results=res1,
                        labels=labels,
                        color_list=cl,
                        coordinates=coords
                    )
                    fr1 = np.array(fr1)
                else:
                    fr1 = res1[0].orig_img[:, :, ::-1].copy()

                if fr1.shape[:2] != (h, w):
                    fr1 = cv2.resize(fr1, (w, h))

                if tracker_2.current_id:
                    tracker_id = tracker_2.current_id
                    coords = tracker_2.current_boxes
                    labels = [
                        f"# {tracker_id[tr]} {models.get('model_2').model.names[coords[tr][-1]]} {coords[tr][-2]:0.2f}"
                        for tr in range(len(tracker_id))
                    ]

                    if len(labels) > 1:
                        cl = colors * len(labels)
                    else:
                        cl = colors

                    fr2 = tracker_2.put_box_on_image(
                        save_path=None,
                        results=res2,
                        labels=labels,
                        color_list=cl,
                        coordinates=coords
                    )
                    fr2 = np.array(fr2)
                else:
                    fr2 = res2[0].orig_img[:, :, ::-1].copy()
                if fr2.shape[:2] != (h, w):
                    fr2 = cv2.resize(fr2, (w, h))

                fr = np.concatenate((fr1, fr2), axis=0)
                headline = f"Обнаружено ковров: {cur_count + len(patterns)}"
                fr = add_headline_to_cv_image(
                    image=fr,
                    headline=headline
                )

                cv_img = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                out.write(cv_img)
            tracker_1.current_id = []
            tracker_1.current_boxes = []
            tracker_2.current_id = []
            tracker_2.current_boxes = []

            # print("time save_path:", time_converter(time.time() - x))
            print("frame time:", time_converter(time.time() - fr_time), '\n')
            logger.info(f"-- patterns: {cur_count + len(patterns)}")
            if i >= finish - 1:
                if save_path:
                    out.release()
                break

    print("===================================================")
    for k in tracker_1.tracker_dict.keys():
        print('tracker_1.tracker_dict', k, tracker_1.tracker_dict[k]['frame_id'])
        coords = [[int(c) for c in box[:4]] for box in tracker_1.tracker_dict[k]['coords']]
        print('tracker_1.tracker_dict', k, coords)
        print()
    print("===================================================")
    for k in tracker_2.tracker_dict.keys():
        print('tracker_2.tracker_dict', k, tracker_2.tracker_dict[k]['frame_id'])
        coords = [[int(c) for c in box[:4]] for box in tracker_2.tracker_dict[k]['coords']]
        print('tracker_2.tracker_dict', k, coords)
        print()
    print("===================================================")
    for i, p in enumerate(patterns):
        print('pattern', i, p)
    # print('tracker_1.tracker_dict', tracker_1.tracker_dict)
    # print('tracker_2.tracker_dict', tracker_2.tracker_dict)
    # print('pattern', patterns)
    # print('cur_bb_1', cur_bb_1)
    # print('cur_bb_2', cur_bb_2)
    # print('true_bb_1', true_bb_1)
    # print('true_bb_2', true_bb_2)
    path = '/media/deny/Новый том/AI/CarpetTracker/tests'
    # save_data(data=tracker_1.tracker_dict, file_path=path, filename='tracker_1_dict')
    # save_data(data=tracker_2.tracker_dict, file_path=path, filename='tracker_2_dict')
    # save_data(data=patterns, file_path=path, filename='patterns')
    # save_data(data=true_bb_1, file_path=path, filename='true_bb_1')
    # save_data(data=true_bb_2, file_path=path, filename='true_bb_2')

    return len(patterns)


def train(weights='yolo8/yolov8n.pt', config='data_custom.yaml', epochs=50, batch_size=4, name=None):
    model = YOLO(weights)
    model.train(data=config, epochs=epochs, batch=batch_size, name=name)


if __name__ == '__main__':
    TRAIN = True
    # train(epochs=100, weights='yolo8/yolov8n.pt', config='data_custom_CAM1.yaml', name='camera_1_mix_l+_8n_100ep')
    # train(epochs=100, weights='yolo8/yolov8n.pt', config='data_custom_CAM2.yaml', name='camera_2_mix_l+_8n_100ep')
    # train(epochs=50, weights='runs/detect/camera_1_mix++_8n_150ep/weights/best.pt', config='data_custom_CAM1.yaml',
    #       name='camera_1_mix++_8n_200ep')
    # train(epochs=50, weights='runs/detect/camera_2_mix++_8n_150ep/weights/best.pt', config='data_custom_CAM2.yaml',
    #       name='camera_2_mix++_8n_200ep')

    # train(epochs=50, weights='yolo8/yolov8n.pt', config='data_custom.yaml')
    # train(epochs=50, weights='yolo8/yolov8s.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8s')
    # train(epochs=50, weights='yolo8/yolov8m.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8m')
    # train(epochs=50, weights='yolo8/yolov8l.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8l')
    # train(epochs=50, weights='yolo8/yolov8x.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8x')

    # train(epochs=50, weights='yolo8/yolov8s.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8s')
    # train(epochs=50, weights='yolo8/yolov8m.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8m')
    # train(epochs=50, weights='yolo8/yolov8l.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8l')
    # train(epochs=50, weights='yolo8/yolov8x.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8x')

    # train(epochs=50, weights='yolo8/yolov8s.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8s')
    # train(epochs=50, weights='yolo8/yolov8m.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8m')
    # train(epochs=50, weights='yolo8/yolov8l.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8l')
    # train(epochs=50, weights='yolo8/yolov8x.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8x')

    PREDICT_IMAGE = True
    # model1 = YOLO('runs/detect/train_camera1/weights/best.pt')
    # img_path='datasets/DataSetMat_Yolo/Olesya/images/KUP_20-21-frame-0-03-41.44.jpg'
    # # img_path='datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01/batch_01_001932.jpg'
    # res = model1(img_path)
    # print(res[0].boxes)
    # print(res[0].orig_shape)
    # img = Image.open(img_path)
    # print(img.size[::-1])

    PREDICT_VIDEO = True
    # model1 = YOLO('runs/detect/camera_1_mix_s+_8n_100ep/weights/best.pt')
    # test_vid = [
    #     # 'videos/sync_test/test 1_cam 1_sync.mp4',
    #     # 'videos/sync_test/test 2_cam 1_sync.mp4',
    #     # 'videos/sync_test/test 3_cam 1_sync.mp4',
    #     # 'videos/sync_test/test 4_cam 1_sync.mp4',
    #     'videos/sync_test/test 5_cam 1_sync.mp4',
    # ]
    # for i, l in enumerate(test_vid):
    #     n = l.split('/')[-1]
    #     detect_video(
    #         model=model1,
    #         video_path=l,
    #         save_path=f'temp/pred_{n}',
    #         remove_perimeter_boxes=True
    #     )
    # model2 = YOLO('runs/detect/camera_2_mix_s+_8n_100ep/weights/best.pt')
    # test_vid_2 = [
    #     # 'videos/sync_test/test 1_cam 2_sync.mp4',
    #     # 'videos/sync_test/test 2_cam 2_sync.mp4',
    #     # 'videos/sync_test/test 3_cam 2_sync.mp4',
    #     # 'videos/sync_test/test 4_cam 2_sync.mp4',
    #     'videos/sync_test/test 5_cam 2_sync.mp4',
    # ]
    # for i, l in enumerate(test_vid_2):
    #     n = l.split('/')[-1]
    #     detect_video(
    #         model=model2,
    #         video_path=l,
    #         save_path=f'temp/pred_{n}'
    #     )

    PREDICT_SYNCH_VIDEO = True
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
        # (model2, "(mix++ 150ep, F% Acc% Sen%)"),
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
        {
            'model_1': 'videos/sync_test/test 14_cam 1_sync.mp4',
            'model_2': 'videos/sync_test/test 14_cam 2_sync.mp4',
            'save_path': 'temp/test 14.mp4',
            'true_count': 157
        },
        {
            'model_1': 'videos/sync_test/test 15_cam 1_sync.mp4',
            'model_2': 'videos/sync_test/test 15_cam 2_sync.mp4',
            'save_path': 'temp/test 15.mp4',
            'true_count': 143
        },
        # {
        #     'model_1': 'videos/short/test 12_cam1_sync.mp4',
        #     'model_2': 'videos/short/test 12_cam2_sync.mp4',
        # },
        # {
        #     'model_1': 'videos/short/test 15_cam 1_sync_16s_20s.mp4',
        #     'model_2': 'videos/short/test 15_cam 2_sync_16s_20s.mp4',
        # },
    ]
    # msg = ''
    for mod in models:
        for i in range(len(video_paths)):
            for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                args = {
                    'conf': c, 'iou': 0., 'speed_limit': SPEED_LIMIT_PERCENT,
                    'start_frame': 0, 'end_frame': 0,
                    'IMAGE_IRRELEVANT_SPACE_PERCENT': IMAGE_IRRELEVANT_SPACE_PERCENT,
                    'MIN_OBJ_SEQUENCE': MIN_OBJ_SEQUENCE, 'MIN_EMPTY_SEQUENCE': MIN_EMPTY_SEQUENCE,
                }
                st = time.time()
                sp = f"{video_paths[i].get('save_path').split('.')[0]} {mod[1]}.mp4" \
                    if video_paths[i].get('save_path') else None
                # sp = ''
                pred_count = detect_synchro_video(
                    models=mod[0],
                    video_paths=video_paths[i],
                    save_path=sp,
                    start=args['start_frame'],
                    finish=args['end_frame'],
                    speed_limit=args['speed_limit'],
                    iou=args['iou'],
                    conf=args['conf']
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

    CREATE_CLASSIFICATION_VIDEO = True
    # models = {
    #         'model_1': YOLO('runs/detect/camera_1_mix+_8n_100ep/weights/best.pt'),
    #         'model_2': YOLO('runs/detect/camera_2_mix+_8n_100ep/weights/best.pt')
    # }
    # csv_files = [
    #     'videos/classification_videos/13-05 ВО.csv',
    #     'videos/classification_videos/16-10 ЦП.csv',
    #     'videos/classification_videos/МОС 19-40.csv',
    #     'videos/classification_videos/Ночь 20-11.csv',
    # ]
    # video_links = [
    #     ['videos/sync_test/13-05 ВО_cam1_sync.mp4', 'videos/sync_test/13-05 ВО_cam2_sync.mp4'],
    #     ['videos/sync_test/16-10 ЦП_cam1_sync.mp4', 'videos/sync_test/16-10 ЦП_cam2_sync.mp4'],
    #     ['videos/sync_test/МОС 19-40_cam1_sync.mp4', 'videos/sync_test/МОС 19-40_cam2_sync.mp4'],
    #     ['videos/sync_test/Ночь 20-11_cam1_sync.mp4', 'videos/sync_test/Ночь 20-11_cam2_sync.mp4'],
    # ]
    # DatasetProcessing.video_class_dataset(
    #     csv_files=csv_files,
    #     video_links=video_links,
    #     save_folder='datasets/class_videos',
    #     yolo_models=models,
    #     box_save_folder='datasets/class_boxes'
    # )
