import copy
import os
import time

import cv2
import numpy as np
import wget
from ultralytics import YOLO
# from tracker import PolyTracker

from parameters import POLY_CAM1_IN, POLY_CAM1_OUT, POLY_CAM2_OUT, POLY_CAM2_IN, ROOT_DIR
from tests.test_tracker import PolyTracker
from utils import get_colors, add_headline_to_cv_image, logger, time_converter, save_data, \
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
    vc = cv2.VideoCapture()
    vc.open(video_path)
    f = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if camera == 1:
        polygon_in = [[int(p[0] * w), int(p[1] * h)] for p in POLY_CAM1_IN]
        polygon_out = [[int(p[0] * w), int(p[1] * h)] for p in POLY_CAM1_OUT]
    if camera == 2:
        polygon_in = [[int(p[0] * w), int(p[1] * h)] for p in POLY_CAM2_IN]
        polygon_out = [[int(p[0] * w), int(p[1] * h)] for p in POLY_CAM2_OUT]
    tracker = PolyTracker(polygon_in=polygon_in, polygon_out=polygon_out, name='mono camera')
    # model = YOLO(model_path)

    if debug:
        print(f"Video data: frames={f}, fps={fps}, width={w}, height={h}")
    if save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (w, h))

    finish = int(f) if finish == 0 or finish < start else finish
    true_bb = {}
    frames = []
    count = 0
    for i in range(0, finish):
        fr_time = time.time()
        _, frame = vc.read()
        frames.append(frame)

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
                # polygon_in=polygon_in,
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
    vc.release()
    cv2.destroyAllWindows()
    return true_bb


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
    range_1 = [(i, round(i * 1000 / fps1, 1)) for i in range(int(f1)+10)]
    range_2 = [(i, round(i * 1000 / fps2, 1)) for i in range(int(f2)+10)]
    (min_range, max_range) = (range_1, range_2) if step == fps1 else (range_2, range_1)
    (min_vc, max_vc) = (vc1, vc2) if step == fps1 else (vc2, vc1)

    polygon_in_1 = [[int(p[0] * w1), int(p[1] * h1)] for p in POLY_CAM1_IN]
    polygon_out_1 = [[int(p[0] * w1), int(p[1] * h1)] for p in POLY_CAM1_OUT]
    tracker_1 = PolyTracker(polygon_in=polygon_in_1, polygon_out=polygon_out_1, name='camera 1')
    polygon_in_2 = [[int(p[0] * w2), int(p[1] * h2)] for p in POLY_CAM2_IN]
    polygon_out_2 = [[int(p[0] * w2), int(p[1] * h2)] for p in POLY_CAM2_OUT]
    tracker_2 = PolyTracker(polygon_in=polygon_in_2, polygon_out=polygon_out_2, name='camera 2')

    def get_closest_id(x: float, data: list[tuple, ...]) -> int:
        dist = [(abs(data[i][1] - x), i) for i in range(len(data))]
        dist = sorted(dist)
        # print("Dist", dist)
        return dist[0][1]

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (w, h * 2))

    # f = min([f1, f2])
    f = f1 if step == fps1 else f2
    finish = int(f) if finish == 0 or finish < start else finish
    true_bb_1, true_bb_2 = [], []
    count = 0
    classes = ['115x200', '115x400', '150x300', '60x90', '85x150']
    class_counter = {cl: 0 for cl in classes}
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

            existing_tracks = [len(tracker_1.track_list), len(tracker_2.track_list)]
            count, last_track_seq = PolyTracker.combine_count(
                count=count,
                last_track_seq=last_track_seq,
                tracker_1_count_frames=copy.deepcopy(tracker_1.count_frames),
                tracker_2_count_frames=copy.deepcopy(tracker_2.count_frames),
                frame_id=i,
                # model=None,
                # class_counter=class_counter,
                # class_list=classes,
                # existing_tracks=existing_tracks
            )
            # save_time = time.time()
            if save_path:
                frame_1 = PolyTracker.prepare_image(
                    image=frame1,
                    colors=colors,
                    tracker_current_boxes=tracker_1.current_boxes,
                    # polygon_in=tracker_1.polygon_in,
                    polygon_out=tracker_1.polygon_out,
                    poly_width=5,
                    reshape=(w, h)
                )

                # frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
                frame_2 = PolyTracker.prepare_image(
                    image=frame2,
                    colors=colors,
                    tracker_current_boxes=tracker_2.current_boxes,
                    # polygon_in=tracker_2.polygon_in,
                    polygon_out=tracker_2.polygon_out,
                    poly_width=2,
                    reshape=(w, h)
                )
                img = np.concatenate((frame_1, frame_2), axis=0)

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
    pass
