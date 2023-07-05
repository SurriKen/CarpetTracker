import copy
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
from tracker import Tracker, PolyTracker
from parameters import SPEED_LIMIT_PERCENT, IMAGE_IRRELEVANT_SPACE_PERCENT, MIN_OBJ_SEQUENCE, MIN_EMPTY_SEQUENCE, \
    POLY_CAM1_IN, POLY_CAM1_OUT, POLY_CAM2_OUT, POLY_CAM2_IN, ROOT_DIR
from utils import get_colors, load_data, add_headline_to_cv_image, logger, time_converter, save_data, \
    clean_diff_image, save_txt

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


def process_image_for_predict(image: np.ndarray, last_image: np.ndarray, mode: str = 'standard'):
    if mode == 'standard':
        return image
    if mode == 'diff':
        diff1 = image - last_image
        cleaned_image, mask = clean_diff_image(diff1, high_color=245)
        return cleaned_image
    if mode == 'masked':
        diff1 = image - last_image
        cleaned_image, mask = clean_diff_image(diff1, high_color=245)
        b_img = np.where(mask == 0, image, (0, 0, 0))
        b_img = b_img.astype(np.uint8)
        b_img = cv2.addWeighted(b_img, 0.4, image, 0.6, 0)
        return b_img
    if mode == 'red':
        diff1 = image - last_image
        cleaned_image, mask = clean_diff_image(diff1, high_color=245)
        r_img = np.where(mask == 0, (0, 0, 255), (255, 0, 0))
        r_img = r_img.astype(np.uint8)
        r_img = cv2.addWeighted(r_img, 0.2, image, 0.8, 0)
        return r_img


def load_kmeans_model(path, name, dict_=False):
    with open(f"{path}/{name}.pkl", "rb") as f:
        model = pickle.load(f)
    lbl_dict = {}
    if dict_:
        lbl_dict = load_data(pickle_path=f"{path}/{name}.dict")
    return model, lbl_dict


def detect_mono_video_polygon(
        model: YOLO,
        camera: int,
        video_path: str,
        save_path: str,
        start: int = 0,
        finish: int = 0,
        iou: float = 0.0,
        conf: float = 0.3,
        interactive_video: bool = False,
        save_boxes_path: str = None,
        save_boxes_mode: str = 'separate',  # single_file
        debug: bool = False,
):
    """
    Detect two synchronized videos and save them as one video with boxes to save_path.

    Args:
        video_path:
        model_path:
        save_boxes_mode:
        save_boxes_path:
        conf:
        iou:
        finish:
        start:
        interactive_video:
        save_path: save_path

    Returns:

    """
    # Get names and colors
    names = ['carpet']
    colors = get_colors(names)

    if camera == 1:
        polygon_in = POLY_CAM1_IN
        polygon_out = POLY_CAM1_OUT
    if camera == 2:
        polygon_in = POLY_CAM2_IN
        polygon_out = POLY_CAM2_OUT
    tracker = PolyTracker(polygon_in=polygon_in, polygon_out=polygon_out, name='mono camera')
    # model = YOLO(model_path)
    vc = cv2.VideoCapture()
    vc.open(video_path)
    f = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if debug:
        print(f"Video data: frames={f}, fps={fps}, width={w}, height={h}")
    if save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (w, h * 2))

    finish = int(f) if finish == 0 or finish < start else finish
    true_bb = {}
    count = 0
    for i in range(0, finish):
        fr_time = time.time()
        _, frame = vc.read()

        if i >= start:
            res = model.predict(frame, iou=iou, conf=conf)
            tracker.process(frame_id=i, boxes=res[0].boxes.data.tolist(), img_shape=res[0].orig_shape[:2], debug=debug)
            if debug:
                logger.info(f"Processed {i + 1} / {f} frames")
                print("track_list", tracker.track_list)
                print("current_boxes", tracker.current_boxes)

            for track in tracker.track_list:
                true_bb[track.get('id')] = [track.get('boxes'), track.get('frame_id')]

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

            if (i + 1) % 100 == 0:
                logger.info(f"Frames {i + 1} / {finish} was processed")
            if save_path:
                out.write(img)
            if debug:
                print("frame time:", time_converter(time.time() - fr_time), '\n')
                logger.info(f"-- count: {count}")

    if save_path:
        out.release()
    if save_boxes_path and save_boxes_mode == 'separate':
        for i, boxes in enumerate(true_bb):
            txt = ''
            for coord in boxes:
                txt = f"{txt}{int(coord)} "
            txt = f"{txt[:-1]}\n"
            save_txt(txt=txt[:-2], txt_path=os.path.join(save_boxes_path, f"{i}.txt"))
    if save_boxes_path and save_boxes_mode == 'single_file':
        save_data(data=true_bb, folder_path=save_boxes_path,
                  filename=f"true_bb_2_{video_path.split('/')[-1].split('_')[0]}")
    return true_bb


def detect_synchro_video(
        models: dict,
        video_paths: dict,
        save_path: str,
        remove_perimeter_boxes=None,
        start: int = 0,
        finish: int = 0,
        speed_limit: float = SPEED_LIMIT_PERCENT,
        iou: float = 0.3,
        conf: float = 0.5,
        draw_polygon: bool = False,
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
                fr1 = res1[0].orig_img[:, :, ::-1].copy()
                if draw_polygon:
                    fr1 = DatasetProcessing.draw_polygons(image=fr1, polygons=POLY_CAM1_IN, outline=(0, 200, 0),
                                                          width=5)
                    fr1 = DatasetProcessing.draw_polygons(image=fr1, polygons=POLY_CAM1_OUT, outline=(200, 0, 0),
                                                          width=5)
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
                        image=fr1,
                        labels=labels,
                        color_list=cl,
                        coordinates=coords,
                        # camera=1,
                        # draw_poly=draw_polygon
                    )
                    fr1 = np.array(fr1)
                # else:
                #     # fr1 = Image.fromarray(image)
                #     # poly = [POLY_CAM1_IN, POLY_CAM1_OUT] if camera == 1 else [POLY_CAM2_IN, POLY_CAM2_OUT]
                #     # image = DatasetProcessing.draw_polygons(polygons=poly[0], image=image, outline='green')
                #     # image = DatasetProcessing.draw_polygons(polygons=poly[1], image=image, outline='red')
                #     # # image.show()
                #     # image = np.array(image)
                #     fr1 = res1[0].orig_img[:, :, ::-1].copy()

                if fr1.shape[:2] != (h, w):
                    fr1 = cv2.resize(fr1, (w, h))

                fr2 = res2[0].orig_img[:, :, ::-1].copy()
                if draw_polygon:
                    fr2 = DatasetProcessing.draw_polygons(image=fr2, polygons=POLY_CAM2_IN, outline=(0, 200, 0),
                                                          width=2)
                    fr2 = DatasetProcessing.draw_polygons(image=fr2, polygons=POLY_CAM2_OUT, outline=(200, 0, 0),
                                                          width=2)
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
                        image=fr2,
                        labels=labels,
                        color_list=cl,
                        coordinates=coords,
                        # camera=2,
                        # draw_poly=draw_polygon
                    )
                    fr2 = np.array(fr2)
                # else:
                #     fr2 = res2[0].orig_img[:, :, ::-1].copy()
                if fr2.shape[:2] != (h, w):
                    fr2 = cv2.resize(fr2, (w, h))

                fr = np.concatenate((fr1, fr2), axis=0)
                headline = f"Обнаружено ковров: {cur_count + len(patterns)}"
                fr = add_headline_to_cv_image(
                    image=fr,
                    headline=headline
                )
                # fr = Image.fromarray(fr)
                # fr.show()

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
                # if i >= 1500:
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


def detect_synchro_video_polygon(
        models: tuple[dict, str],
        video_paths: dict,
        save_path: str,
        start: int = 0,
        finish: int = 0,
        iou: float = 0.3,
        conf: float = 0.5,
        interactive_video: bool = False,
        mode: str = 'standard',  # standard, diff, masked, red
        save_boxes: bool = False,
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
    names = ['carpet']
    colors = get_colors(names)

    tracker_1 = PolyTracker(polygon_in=POLY_CAM1_IN, polygon_out=POLY_CAM1_OUT, name='camera 1')
    tracker_2 = PolyTracker(polygon_in=POLY_CAM2_IN, polygon_out=POLY_CAM2_OUT, name='camera 2')

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
    print(f"fps1={fps1}, fps2={fps2}\n")

    w = min([w1, w2])
    h = min([h1, h2])

    step = min([fps1, fps2])
    range_1 = [(i, round(i * 1000 / fps1, 1)) for i in range(int(f1))]
    range_2 = [(i, round(i * 1000 / fps2, 1)) for i in range(int(f2))]
    (min_range, max_range) = (range_1, range_2) if step == fps1 else (range_2, range_1)
    (min_vc, max_vc) = (vc1, vc2) if step == fps1 else (vc2, vc1)

    def get_closest_id(x: float, data: list[tuple, ...]) -> int:
        dist = [(abs(data[i][1] - x), i) for i in range(len(data))]
        dist = sorted(dist)
        # print("Dist", dist)
        return dist[0][1]

    if save_path and mode in ['standard', 'masked', 'red']:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (w, h * 2))
    if save_path and mode in ['diff']:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (w * 2, h * 2))

    # f = min([f1, f2])
    f = f1 if step == fps1 else f2
    finish = int(f) if finish == 0 or finish < start else finish
    true_bb_1, true_bb_2 = [], []
    count = 0
    last_track_seq = {'tr1': [], 'tr2': []}
    last_img_1, last_img_2 = [], []
    for i in range(0, finish):
        fr_time = time.time()
        _, frame1 = min_vc.read()

        closest_id = get_closest_id(min_range[0][1], max_range[:10])
        min_range.pop(0)
        ids = list(range(closest_id)) if closest_id else [0]
        ids = sorted(ids, reverse=True)
        for id in ids:
            max_range.pop(id)
            _, frame2 = max_vc.read()
        # logger.info(f'-- min_range {min_range[:5]}, \nmax_range {max_range[:5]}')

        if i >= start and len(last_img_1) and len(last_img_2):
            logger.info(f"Processed {i + 1} / {f} frames")
            # image_1_time = time.time()
            image_1 = process_image_for_predict(image=frame1, last_image=last_img_1, mode=mode)
            # print("image_1_time process", time_converter(time.time() - image_1_time))
            res1 = models[0].get('model_1').predict(image_1, iou=iou, conf=conf)
            true_bb_1.append(res1[0].boxes.data.tolist())
            tracker_1.process(frame_id=i, boxes=res1[0].boxes.data.tolist(), img_shape=res1[0].orig_shape[:2],
                              debug=False)

            # image_2_time = time.time()
            image_2 = process_image_for_predict(image=frame2, last_image=last_img_2, mode=mode)
            # print("image_2_time process", time_converter(time.time() - image_2_time))
            res2 = models[0].get('model_2').predict(image_2, iou=iou, conf=conf)
            true_bb_2.append(res2[0].boxes.data.tolist())
            tracker_2.process(frame_id=i, boxes=res2[0].boxes.data.tolist(), img_shape=res2[0].orig_shape[:2],
                              debug=False)

            count, last_track_seq = PolyTracker.combine_count(
                count=count,
                last_track_seq=last_track_seq,
                tracker_1_count_frames=copy.deepcopy(tracker_1.count_frames),
                tracker_2_count_frames=copy.deepcopy(tracker_2.count_frames),
                frame_id=i
            )
            # save_time = time.time()
            if save_path:
                frame_1 = PolyTracker.prepare_image(
                    image=frame1,
                    colors=colors,
                    tracker_current_boxes=tracker_1.current_boxes,
                    polygon_in=POLY_CAM1_IN,
                    polygon_out=POLY_CAM1_OUT,
                    poly_width=5,
                    reshape=(w, h)
                )

                # frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
                frame_2 = PolyTracker.prepare_image(
                    image=frame2,
                    colors=colors,
                    tracker_current_boxes=tracker_2.current_boxes,
                    polygon_in=POLY_CAM2_IN,
                    polygon_out=POLY_CAM2_OUT,
                    poly_width=2,
                    reshape=(w, h)
                )
                if mode in ['diff', 'masked', 'red']:
                    image_1 = PolyTracker.prepare_image(
                        image=image_1,
                        colors=colors,
                        tracker_current_boxes=tracker_1.current_boxes,
                        polygon_in=POLY_CAM1_IN,
                        polygon_out=POLY_CAM1_OUT,
                        poly_width=5,
                        reshape=(w, h)
                    )
                    image_2 = PolyTracker.prepare_image(
                        image=image_2,
                        colors=colors,
                        tracker_current_boxes=tracker_2.current_boxes,
                        polygon_in=POLY_CAM2_IN,
                        polygon_out=POLY_CAM2_OUT,
                        poly_width=2,
                        reshape=(w, h)
                    )
                img_process = time.time()
                if save_path and mode == 'standard':
                    img = np.concatenate((frame_1, frame_2), axis=0)
                if save_path and mode in ['diff']:
                    img1 = np.concatenate((frame_1, frame_2), axis=0)
                    img2 = np.concatenate((image_1, image_2), axis=0)
                    img = np.concatenate((img1, img2), axis=1)
                if save_path and mode in ['masked', 'red']:
                    img = np.concatenate((image_1, image_2), axis=0)

                headline = f"Обнаружено ковров: {count}\nТрекер 1: {tracker_1.count}\nТрекер 2: {tracker_2.count}"
                img = add_headline_to_cv_image(
                    image=img,
                    headline=headline
                )
                if interactive_video:
                    cv2.imshow('1', img)
                    cv2.waitKey(1)

                if (i + 1) % 100 == 0:
                    logger.info(f"Frames {i + 1} / {finish} was processed")
                out.write(img)

            # print("time save_path:", time_converter(time.time() - x))
            print("frame time:", time_converter(time.time() - fr_time),
                  # "save_time", time_converter(time.time() - save_time),
                  # "img_process", time_converter(img_process - save_time),
                  '\n')
            logger.info(f"-- count: {count}")
            if i >= finish - 1 or not min_range or not max_range:
                break

        last_img_1 = frame1
        last_img_2 = frame2
    if save_path:
        out.release()
    if save_boxes:
        path = os.path.join(ROOT_DIR, 'tests/boxes')
        save_data(data=true_bb_1, folder_path=path,
                  filename=f"true_bb_1_{video_paths.get('model_1').split('/')[-1].split('_')[0]} {models[1]}")
        save_data(data=true_bb_2, folder_path=path,
                  filename=f"true_bb_2_{video_paths.get('model_2').split('/')[-1].split('_')[0]} {models[1]}")
    return count


def train(weights='yolo_weights/yolov8n.pt', config='data_custom.yaml', epochs=50, batch_size=4, name=None):
    model = YOLO(weights)
    model.train(data=config, epochs=epochs, batch=batch_size, name=name)


if __name__ == '__main__':
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
    # model1 = {
    #     'model_1': YOLO('runs/detect/camera_1_mix+_8n_100ep/weights/best.pt'),
    #     'model_2': YOLO('runs/detect/camera_2_mix+_8n_100ep/weights/best.pt')
    # }
    # model2 = {
    #     'model_1': YOLO('runs/detect/camera_1_mix++_8n_150ep/weights/best.pt'),
    #     'model_2': YOLO('runs/detect/camera_2_mix++_8n_150ep/weights/best.pt')
    # }
    # models = [
    #     (model1, "(mix+ 100ep, F% Acc% Sen%)"),
    #     # (model2, "(mix++ 150ep, F% Acc% Sen%)"),
    # ]
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
    #     # {
    #     #     'model_1': 'videos/sync_test/test 5_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 5_cam 2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 6_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 6_cam 2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 7_cam1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 7_cam2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 8_cam1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 8_cam2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 10_cam1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 10_cam2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 11_cam1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 11_cam2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 12_cam1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 12_cam2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 13_cam1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 13_cam2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 14_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 14_cam 2_sync.mp4',
    #     #     'save_path': 'temp/test 14.mp4',
    #     #     'true_count': 157
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 15_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 15_cam 2_sync.mp4',
    #     #     'save_path': 'temp/test 15.mp4',
    #     #     'true_count': 143
    #     # },
    #     {
    #         'model_1': 'videos/sync_test/test 16_cam 1_sync.mp4',
    #         'model_2': 'videos/sync_test/test 16_cam 2_sync.mp4',
    #         'save_path': 'temp/test 16.mp4',
    #         'true_count': 168
    #     },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 17_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 17_cam 2_sync.mp4',
    #     #     'save_path': 'temp/test 17.mp4',
    #     #     'true_count': 167
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 18_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 18_cam 2_sync.mp4',
    #     #     'save_path': 'temp/test 18.mp4',
    #     #     'true_count': 129
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 19_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 19_cam 2_sync.mp4',
    #     #     'save_path': 'temp/test 19.mp4',
    #     #     'true_count': 136
    #     # },
    #     # {
    #     #     'model_1': 'videos/sync_test/test 20_cam 1_sync.mp4',
    #     #     'model_2': 'videos/sync_test/test 20_cam 2_sync.mp4',
    #     #     'save_path': 'temp/test 20.mp4',
    #     #     'true_count': 139
    #     # },
    #     # {
    #     #     'model_1': 'videos/short/test 12_cam1_sync.mp4',
    #     #     'model_2': 'videos/short/test 12_cam2_sync.mp4',
    #     # },
    #     # {
    #     #     'model_1': 'videos/short/test 15_cam 1_sync_16s_20s.mp4',
    #     #     'model_2': 'videos/short/test 15_cam 2_sync_16s_20s.mp4',
    #     # },
    # ]
    # # msg = ''
    # for mod in models:
    #     for i in range(len(video_paths)):
    #         # for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #         args = {
    #             'conf': 0.3, 'iou': 0., 'speed_limit': SPEED_LIMIT_PERCENT,
    #             'start_frame': 0, 'end_frame': 0,
    #             'IMAGE_IRRELEVANT_SPACE_PERCENT': IMAGE_IRRELEVANT_SPACE_PERCENT,
    #             'MIN_OBJ_SEQUENCE': MIN_OBJ_SEQUENCE, 'MIN_EMPTY_SEQUENCE': MIN_EMPTY_SEQUENCE,
    #             'draw_polygon': True
    #         }
    #         st = time.time()
    #         sp = f"{video_paths[i].get('save_path').split('.')[0]} {mod[1]}.mp4" \
    #             if video_paths[i].get('save_path') else None
    #         # sp = ''
    #         pred_count = detect_synchro_video(
    #             models=mod[0],
    #             video_paths=video_paths[i],
    #             save_path=sp,
    #             start=args['start_frame'],
    #             finish=args['end_frame'],
    #             speed_limit=args['speed_limit'],
    #             iou=args['iou'],
    #             conf=args['conf'],
    #             draw_polygon=args['draw_polygon']
    #         )
    #         dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    #         txt = f"{dt} =========== Predict is finished ===========\n" \
    #               f"- Model {mod[1]}\n" \
    #               f"- Video '{video_paths[i]}'\n" \
    #               f"- True count: '{video_paths[i].get('true_count')}; Predict count: '{pred_count}'\n" \
    #               f"- Saves as '{sp}'\n" \
    #               f"- Predict args: {args}\n" \
    #               f"- Process time: {time_converter(time.time() - st)}\n"
    #         logger.info(f"Predict is finished. Model {mod[1]}. Video {video_paths[i].get('save_path')}")
    #
    #         msg = f"{dt}   {txt}\n\n"
    #         save_txt(txt=msg, txt_path='logs/predict_synch_log.txt', mode='a')

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
