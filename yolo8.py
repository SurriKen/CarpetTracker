import os
import pickle
import shutil

import cv2
import numpy as np
import wget
from ultralytics import YOLO

from dataset_processing import DatasetProcessing
from tracker import Tracker
from parameters import SPEED_LIMIT_PERCENT
from utils import get_colors, load_dict, add_headline_to_cv_image, logger

yolov8_types = {
    "yolov8n": {"Test Size": 640, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"},
    "yolov8s": {"Test Size": 640, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"},
    "yolov8m": {"Test Size": 1280, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"},
    "yolov8I": {"Test Size": 1280, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"},
    "yolov8x": {"Test Size": 1280, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"},
}


def load_yolo_v8(v8_mode="yolov8n"):
    # if not os.path.isdir("yolov7"):
    #     os.system("git clone https://github.com/WongKinYiu/yolov7.git")

    if not os.path.isdir(f"yolov8/{v8_mode}.pt"):
        url = yolov8_types[v8_mode]["link"]
        wget.download(url, f"yolov8/{v8_mode}.pt")


def load_kmeans_model(path, name, dict_=False):
    with open(f"{path}/{name}.pkl", "rb") as f:
        model = pickle.load(f)
    lbl_dict = {}
    if dict_:
        lbl_dict = load_dict(pickle_path=f"{path}/{name}.dict")
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
        speed_limit: float = SPEED_LIMIT_PERCENT
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
    for i in range(0, finish):
        ret1, frame1 = vc1.read()
        ret2, frame2 = vc2.read()

        if i >= start:
            logger.info(f"Processed {i + 1} / {f} frames")

            res1 = models.get('model_1')(frame1)
            result = {'boxes': res1[0].boxes.boxes.tolist(), 'orig_shape': res1[0].orig_shape}
            tracker_1.process(
                frame_id=i,
                predict_camera_1=result,
                remove_perimeter_boxes=remove_perimeter_boxes.get('model_1'),
                speed_limit_percent=speed_limit
            )

            res2 = models.get('model_2')(frame2)
            result = {'boxes': res2[0].boxes.boxes.tolist(), 'orig_shape': res2[0].orig_shape}
            tracker_2.process(
                frame_id=i,
                predict_camera_1=result,
                remove_perimeter_boxes=remove_perimeter_boxes.get('model_2'),
                speed_limit_percent=speed_limit
            )

            patterns = Tracker().get_pattern(
                input=Tracker.join_frame_id(
                    tracker_dict_1=tracker_1.tracker_dict,
                    tracker_dict_2=tracker_2.tracker_dict
                )
            )
            logger.info(f"-- patterns: {len(patterns)}")

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
                headline = f"Обнаружено ковров: {len(patterns)}"
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
            if i >= finish - 1:
                out.release()
                break


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
    # models = {
    #     'model_1': YOLO('runs/detect/camera_1_mix_m+_8n_100ep/weights/best.pt'),
    #     'model_2': YOLO('runs/detect/camera_2_mix_m+_8n_100ep/weights/best.pt')
    # }
    # video_paths = [
    #     # {
    #     #     'model_1': 'videos/sync_test/test 1_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 1_cam 2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 2_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 2_cam 2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 3_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 3_cam 2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 4_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 4_cam 2_sync.mp4',
    #     # },
    #     {
    #         'model_1': 'videos/sync_test/test 5_cam 1_sync.mp4',
    #         'model_2': 'videos/sync_test/test 5_cam 2_sync.mp4',
    #     },
    # ]
    # save_path = [
    #     # f'temp/pred_test 1.mp4',
    #     # f'temp/pred_test 2.mp4',
    #     # f'temp/pred_test 3.mp4',
    #     # f'temp/pred_test 4.mp4',
    #     f'temp/pred_test 5.mp4',
    # ]
    # for i in range(len(video_paths)):
    #     detect_synchro_video(
    #         models=models,
    #         video_paths=video_paths[i],
    #         save_path=save_path[i],
    #         start=0,
    #         finish=0,
    #         speed_limit=SPEED_LIMIT_PERCENT
    #     )
    pass
