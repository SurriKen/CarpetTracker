import copy
import os
from collections import Counter
import cv2
import numpy as np
import wget
from ultralytics import YOLO
from classification.nn_classificator import VideoClassifier
from tracker.tracker import PolyTracker
from parameters import *
from utils import get_colors, add_headline_to_cv_image, dict_to_csv, get_name_from_link

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
        v8_mode: str - type of yolo v8 (key in yolov8_types dict)
    """

    if not os.path.isdir(f"yolov8/{v8_mode}.pt"):
        url = yolov8_types[v8_mode]["link"]
        wget.download(url, f"yolov8/{v8_mode}.pt")


def detect_mono_video_polygon(
        model: YOLO,
        camera: int,
        video_path: str,
        save_path: str,
        start: int = 0,
        finish: int = 0,
        interactive_video: bool = False
) -> int:
    """
    Args:
        model: YOLO, trained YOLO model,
        camera: int, id of used camera, 1 for corner camera and 2 for center wall camera
        video_path: str, path to initial video
        save_path: str, path to save predicted video
        start: int, frame to start prediction (0 by default),
        finish: int, frame to end prediction (0 by default),
        interactive_video: bool, show video in interactive mode (False by default)

    Returns:
        int, found object number
    """
    # Get names and colors
    names = ['carpet']
    colors = get_colors(names)
    vc = cv2.VideoCapture()
    vc.open(video_path)
    f = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if camera == 1:
        polygon_in = [[int(p[0] * w), int(p[1] * h)] for p in POLY_CAM1_IN]
        polygon_out = [[int(p[0] * w), int(p[1] * h)] for p in POLY_CAM1_OUT]
    if camera == 2:
        polygon_in = [[int(p[0] * w), int(p[1] * h)] for p in POLY_CAM2_IN]
        polygon_out = [[int(p[0] * w), int(p[1] * h)] for p in POLY_CAM2_OUT]
    tracker = PolyTracker(polygon_in=polygon_in, polygon_out=polygon_out, name='mono camera')

    if save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (w, h))

    finish = int(f) if finish == 0 or finish < start else finish
    frames = []

    for i in range(0, finish):
        _, frame = vc.read()
        frames.append(frame)

        if i >= start:
            res = model.predict(frame, iou=0, conf=0.3)
            tracker.process(frame_id=i, boxes=res[0].boxes.data.tolist(), img_shape=res[0].orig_shape[:2])

            frame = PolyTracker.prepare_image(
                image=frame,
                colors=colors,
                tracker_current_boxes=tracker.current_boxes,
                polygon_in=polygon_in,
                polygon_out=polygon_out,
                poly_width=5,
                reshape=(w, h)
            )

            headline = f"Обнаружено ковров: {tracker.count}"
            img = add_headline_to_cv_image(
                image=frame,
                headline=headline
            )
            if interactive_video:
                cv2.imshow('1', img)
                cv2.waitKey(1)

            if save_path:
                out.write(img)

    if save_path:
        out.release()

    vc.release()
    cv2.destroyAllWindows()
    return tracker.count


def detect_synchro_video_polygon(
        models: dict,
        video_paths: dict,
        save_path: str = '',
        class_model: VideoClassifier = None,
        start: int = 0,
        finish: int = 0,
        interactive_video: bool = False,
        debug: bool = False,
        stream: bool = False
) -> dict:
    """
    Detect two synchronized videos and save them as one video with boxes to save_path.

    Args:
        models: dict, {'model_1': model_1, "model_2": model_2}
        video_paths: dict, {'model_1': path_1, "model_2": path_2}, path_1 MUST be the video path from corner camera,
                        path_2 MUST be the video path from center camera
        save_path: str, path to save predicted video
        class_model: VideoClassifier = None,
        start: int, frame to start prediction (0 by default),
        finish: int, frame to end prediction (0 by default),
        interactive_video: bool, show video in interactive mode (False by default)
        debug: bool = False

    Returns:
        list of predicted classes ordered from start to finish of predicdion
    """

    def get_closest_id(x: float, data: list[tuple, ...]) -> int:
        dist = [(abs(data[i][1] - x), i) for i in range(len(data))]
        dist = sorted(dist)
        return dist[0][1]

    # Get names and colors
    names = ['carpet']
    colors = get_colors(names)
    w = 640
    h = 360
    save_folder = save_path[:-(len(save_path.split('/')[-1]) + 1)]
    filename = get_name_from_link(save_path)

    if not stream:
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

        step = min([fps1, fps2])
        range_1 = [(i, round(i * 1000 / fps1, 1)) for i in range(int(f1 * fps1 / step) + 1)]
        range_2 = [(i, round(i * 1000 / fps2, 1)) for i in range(int(f2 * fps2 / step) + 1)]
        (min_range, max_range) = (range_1, range_2) if step == fps1 else (range_2, range_1)
        (min_vc, max_vc) = (vc1, vc2) if step == fps1 else (vc2, vc1)

        polygon_in_1 = [[int(p[0] * w1), int(p[1] * h1)] for p in POLY_CAM1_IN]
        polygon_out_1 = [[int(p[0] * w1), int(p[1] * h1)] for p in POLY_CAM1_OUT]
        tracker_1 = PolyTracker(polygon_in=polygon_in_1, polygon_out=polygon_out_1, name='camera 1')
        polygon_in_2 = [[int(p[0] * w2), int(p[1] * h2)] for p in POLY_CAM2_IN]
        polygon_out_2 = [[int(p[0] * w2), int(p[1] * h2)] for p in POLY_CAM2_OUT]
        tracker_2 = PolyTracker(polygon_in=polygon_in_2, polygon_out=polygon_out_2, name='camera 2')

        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (w, h * 2))

        f = f1 if step == fps1 else f2
        finish = int(f) - 1 if finish == 0 or finish < start else finish
        count = 0
        class_counter = []
        last_track_seq = {'tr1': [], 'tr2': []}
        last_img_1, last_img_2 = [], []
        tracks = []
        cl_count = {cl: 0 for cl in CLASSES}
        result = dict(total_count=count)
        result.update(cl_count)
        stop_flag = False
        for i in range(0, finish):
            _, frame1 = min_vc.read()

            closest_id = get_closest_id(min_range[0][1], max_range[:10])
            min_range.pop(0)
            ids = list(range(closest_id)) if closest_id else [0]
            ids = sorted(ids, reverse=True)
            for id in ids:
                max_range.pop(id)
                _, frame2 = max_vc.read()

            if i >= start and len(last_img_1) and len(last_img_2):
                if i == finish - 1:
                    stop_flag = True

                res1 = models.get('model_1').predict(frame1, iou=0, conf=0.3)
                tracker_1.process(frame_id=i, boxes=res1[0].boxes.data.tolist(), img_shape=res1[0].orig_shape[:2],
                                  debug=False, stop_flag=stop_flag)

                res2 = models.get('model_2').predict(frame2, iou=0, conf=0.3)
                tracker_2.process(frame_id=i, boxes=res2[0].boxes.data.tolist(), img_shape=res2[0].orig_shape[:2],
                                  debug=False, stop_flag=stop_flag)
                if debug:
                    print('================================================================')
                    print(f"Current_frame = {i}, current count = {count}, stop_flag={stop_flag}")
                    print(f"Input boxes, tracker 1 = {res1[0].boxes.data.tolist()}, tracker 2 = {res2[0].boxes.data.tolist()}")
                    print(f'tracker_1.track_list. Track num = {len(tracker_1.track_list)}')
                    for tr in tracker_1.track_list:
                        print(f"--ID={tr['id']}, frames={tr['frame_id']}, check_out={tr['check_out']}, "
                              f"boxes={tr['boxes']}")
                    print('tracker_1.dead_boxes', tracker_1.dead_boxes)
                    print(f'tracker_2.track_list. Track num = {len(tracker_2.track_list)}')
                    for tr in tracker_2.track_list:
                        print(f"--ID={tr['id']}, frames={tr['frame_id']}, check_out={tr['check_out']}, "
                              f"boxes={tr['boxes']}")

                existing_tracks = [len(tracker_1.track_list), len(tracker_2.track_list)]
                class_counter, last_track_seq, end_track = PolyTracker.combine_count(
                    count=count,
                    last_track_seq=last_track_seq,
                    tracker_1_count_frames=copy.deepcopy(tracker_1.count_frames),
                    tracker_2_count_frames=copy.deepcopy(tracker_2.count_frames),
                    frame_id=i,
                    class_model=class_model,
                    class_counter=class_counter,
                    class_list=CLASSES,
                    existing_tracks=existing_tracks,
                    frame_size=(128, 128),
                    stop_flag=stop_flag,
                    debug=False
                )
                if debug:
                    print('class_counter, last_track_seq, end_track', class_counter, last_track_seq, end_track)
                count = len(class_counter)

                if class_counter:
                    cl_count.update(dict(Counter(class_counter)))
                    result['total_count'] = count
                    result.update(cl_count)
                    dict_to_csv(data=result, folder_path=save_folder, filename=filename)
                if end_track != {'tr1': [], 'tr2': []}:
                    tracks.append(end_track)

                if save_path:
                    frame_1 = PolyTracker.prepare_image(
                        image=frame1,
                        colors=colors,
                        tracker_current_boxes=tracker_1.current_boxes,
                        polygon_in=tracker_1.polygon_in,
                        polygon_out=tracker_1.polygon_out,
                        poly_width=5,
                        reshape=(w, h)
                    )

                    frame_2 = PolyTracker.prepare_image(
                        image=frame2,
                        colors=colors,
                        tracker_current_boxes=tracker_2.current_boxes,
                        polygon_in=tracker_2.polygon_in,
                        polygon_out=tracker_2.polygon_out,
                        poly_width=2,
                        reshape=(w, h)
                    )
                    img = np.concatenate((frame_1, frame_2), axis=0)
                    txt = ''
                    for cl in CLASSES:
                        txt = f"{txt}{cl} - {cl_count.get(cl)}\n"
                    headline = f"Обнаружено ковров: {count}\n" \
                               f"{txt[:-1]}"
                    img = add_headline_to_cv_image(
                        image=img,
                        headline=headline
                    )
                    if interactive_video:
                        cv2.imshow('1', img)
                        cv2.waitKey(1)

                    out.write(img)

                if i >= finish - 1 or not min_range or not max_range:
                    break

            last_img_1 = frame1
            last_img_2 = frame2
        out.release()
        cv2.destroyAllWindows()
        if save_path:
            out.release()
        result = dict(total_count=count)
        result.update(cl_count)
        return result

    else:
        vc1 = cv2.VideoCapture(video_paths.get("model_1"))
        w1 = int(vc1.get(cv2.CAP_PROP_FRAME_WIDTH))
        h1 = int(vc1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vc2 = cv2.VideoCapture(video_paths.get("model_2"))
        w2 = int(vc2.get(cv2.CAP_PROP_FRAME_WIDTH))
        h2 = int(vc2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        polygon_in_1 = [[int(p[0] * w1), int(p[1] * h1)] for p in POLY_CAM1_IN]
        polygon_out_1 = [[int(p[0] * w1), int(p[1] * h1)] for p in POLY_CAM1_OUT]
        tracker_1 = PolyTracker(polygon_in=polygon_in_1, polygon_out=polygon_out_1, name='camera 1')

        polygon_in_2 = [[int(p[0] * w2), int(p[1] * h2)] for p in POLY_CAM2_IN]
        polygon_out_2 = [[int(p[0] * w2), int(p[1] * h2)] for p in POLY_CAM2_OUT]
        tracker_2 = PolyTracker(polygon_in=polygon_in_2, polygon_out=polygon_out_2, name='camera 2')

        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (w, h * 2))

        count = 0
        class_counter = []
        last_track_seq = {'tr1': [], 'tr2': []}
        tracks = []
        frame_id = 0
        cl_count = {cl: 0 for cl in CLASSES}
        result = dict(total_count=count)
        result.update(cl_count)

        while vc1.isOpened() and vc2.isOpened():
            _, frame1 = vc1.read()
            _, frame2 = vc2.read()

            res1 = models.get('model_1').predict(frame1, iou=0, conf=0.3)
            tracker_1.process(frame_id=frame_id, boxes=res1[0].boxes.data.tolist(), img_shape=res1[0].orig_shape[:2],
                              debug=False)

            res2 = models.get('model_2').predict(frame2, iou=0, conf=0.3)
            tracker_2.process(frame_id=frame_id, boxes=res2[0].boxes.data.tolist(), img_shape=res2[0].orig_shape[:2],
                              debug=False)
            if debug:
                print('================================================================')
                print(f"Current_frame = {frame_id}, current count = {count}")
                print(
                    f"Input boxes, tracker 1 = {res1[0].boxes.data.tolist()}, tracker 2 = {res2[0].boxes.data.tolist()}")
                print(f'tracker_1.track_list. Track num = {len(tracker_1.track_list)}')
                for tr in tracker_1.track_list:
                    print(f"--ID={tr['id']}, frames={tr['frame_id']}, check_out={tr['check_out']}, "
                          f"boxes={tr['boxes']}")
                print('tracker_1.dead_boxes', tracker_1.dead_boxes)
                print(f'tracker_2.track_list. Track num = {len(tracker_2.track_list)}')
                for tr in tracker_2.track_list:
                    print(f"--ID={tr['id']}, frames={tr['frame_id']}, check_out={tr['check_out']}, "
                          f"boxes={tr['boxes']}")

            existing_tracks = [len(tracker_1.track_list), len(tracker_2.track_list)]
            class_counter, last_track_seq, end_track = PolyTracker.combine_count(
                count=count,
                last_track_seq=last_track_seq,
                tracker_1_count_frames=copy.deepcopy(tracker_1.count_frames),
                tracker_2_count_frames=copy.deepcopy(tracker_2.count_frames),
                frame_id=frame_id,
                class_model=class_model,
                class_counter=class_counter,
                class_list=CLASSES,
                existing_tracks=existing_tracks,
                frame_size=(128, 128),
                debug=False
            )
            frame_id += 1

            if debug:
                print('class_counter, last_track_seq, end_track', class_counter, last_track_seq, end_track)
            count = len(class_counter)

            if class_counter:
                cl_count.update(dict(Counter(class_counter)))
                result['total_count'] = count
                result.update(cl_count)
                dict_to_csv(data=result, folder_path=save_folder, filename=filename)
            if end_track != {'tr1': [], 'tr2': []}:
                tracks.append(end_track)

            if save_path:
                frame_1 = PolyTracker.prepare_image(
                    image=frame1,
                    colors=colors,
                    tracker_current_boxes=tracker_1.current_boxes,
                    polygon_in=tracker_1.polygon_in,
                    polygon_out=tracker_1.polygon_out,
                    poly_width=5,
                    reshape=(w, h)
                )

                frame_2 = PolyTracker.prepare_image(
                    image=frame2,
                    colors=colors,
                    tracker_current_boxes=tracker_2.current_boxes,
                    polygon_in=tracker_2.polygon_in,
                    polygon_out=tracker_2.polygon_out,
                    poly_width=2,
                    reshape=(w, h)
                )
                img = np.concatenate((frame_1, frame_2), axis=0)
                txt = ''
                for cl in CLASSES:
                    txt = f"{txt}{cl} - {cl_count.get(cl)}\n"
                headline = f"Обнаружено ковров: {count}\n" \
                           f"{txt[:-1]}"
                img = add_headline_to_cv_image(
                    image=img,
                    headline=headline
                )
                if interactive_video:
                    cv2.imshow('1', img)
                    cv2.waitKey(1)

                out.write(img)

        out.release()
        cv2.destroyAllWindows()
        if save_path:
            out.release()
        return result


def train(weights, config, epochs=50, batch_size=4, name=None):
    model = YOLO(weights)
    model.train(data=config, epochs=epochs, batch=batch_size, name=name)


if __name__ == '__main__':
    pass
