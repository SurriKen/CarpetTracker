import os
import shutil

import cv2
import numpy as np
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO

from DatasetProcessing import DatasetProcessing
from parameters import IMAGE_IRRELEVANT_SPACE_PERCENT, MIN_OBJ_SEQUENCE, SPEED_LIMIT_PERCENT, MIN_EMPTY_SEQUENCE
from test import tracker_1_coord, tracker_2_coord
from utils import logger, get_colors, add_headline_to_cv_image

# SPEED_LIMIT_PERCENT = 0.005


class Tracker:
    def __init__(self):
        self.tracker_dict = {}
        self.working_idxs = []
        self.last_relevant_frames = []
        self.empty_seq = []
        self.carpet_seq = []
        self.dead_boxes = {}
        self.count = 0
        self.current_boxes = []
        self.current_id = []
        pass

    @staticmethod
    def put_box_on_image(save_path, results, labels, color_list, coordinates):
        image = results[0].orig_img[:, :, ::-1].copy()
        image = np.transpose(image, (2, 0, 1))
        w, h = image.shape[:2]
        image = torch.from_numpy(image)
        coord = []
        for box in coordinates:
            # box = box.boxes.tolist()[0]
            coord.append([
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
            ])
        bbox = torch.tensor(coord, dtype=torch.int)
        if bbox.tolist():
            image_true = draw_bounding_boxes(
                image, bbox, width=3, labels=labels, colors=color_list, fill=True, font='arial.ttf',
                font_size=int(h * 0.02))
            image = torchvision.transforms.ToPILImage()(image_true)
        else:
            image = torchvision.transforms.ToPILImage()(image)
        if save_path:
            image.save(f'{save_path}')
        return image

    @staticmethod
    def remove_perimeter_boxes(box: list, origin_shape: tuple):
        """
        Remove boxes on the perimeter of the image with IMAGE_IRRELEVANT_SPACE_PERCENT of the image size

        :param box: list of coordinates with first 4 index positions [x1, y1, x2, y2]
        :param origin_shape: shape of the image
        :return: list of coordinates [x1, y1, x2, y2] or empty list if box is not in the image
        """
        # print(box, origin_shape)
        x_min = int(origin_shape[1] * IMAGE_IRRELEVANT_SPACE_PERCENT)
        x_max = int(origin_shape[1] - origin_shape[1] * IMAGE_IRRELEVANT_SPACE_PERCENT)
        y_min = int(origin_shape[0] * IMAGE_IRRELEVANT_SPACE_PERCENT)
        y_max = int(origin_shape[0] - origin_shape[0] * IMAGE_IRRELEVANT_SPACE_PERCENT)
        box_center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        if box_center[0] not in range(x_min, x_max) or box_center[1] not in range(y_min, y_max):
            return []
        else:
            return box

    @staticmethod
    def get_distance(box1: list, box2: list) -> float:
        """
        :param box1: list of coordinates with first 4 index positions [x1, y1, x2, y2]
        :param box2: list of coordinates with first 4 index positions [x1, y1, x2, y2]
        :return: float distance between the two box centers
        """
        # print(box1, box2)
        c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    @staticmethod
    def lists_difference(list1, list2):
        """
        :param list1: list of indexes
        :param list2: list of indexes
        :return: list of difference
        """
        set1 = set(list1)
        set2 = set(list2)
        return list(set1.difference(set2))

    # @staticmethod
    def add_new_id_form(self, new_id):
        """
        Create a new id empty form for new track

        :param new_id: int new id
        """
        self.tracker_dict[new_id] = {
            'frame_id': [],  # 10, 20, 30
            'coords': [],
            # 'speed': [],
            # 'coords_2': [],
            # 'status': 'run',  # 'end'
            # 'images': [],  # arrays from 2 cameras
            # 'carpet_size': '',
            'shift': [],
            'countable': False,  # True
            'relevant': True,  # False
        }
        # return id_dict

    @staticmethod
    def join_frame_id(tracker_dict_1: dict, tracker_dict_2: dict) -> list:
        """
        Get from 2 trackers list of relevant frames id

        :param tracker_dict_1: dict with frame_id as key
        :param tracker_dict_2: dict with frame_id as key
        :return: list of unique frame_id
        """
        pattern = []
        for k in tracker_dict_1.keys():
            pattern.extend(tracker_dict_1[k]['frame_id'])
        for k in tracker_dict_2.keys():
            pattern.extend(tracker_dict_2[k]['frame_id'])
        return sorted(list(set(pattern)))

    @staticmethod
    def get_pattern(input: list) -> list:
        """
        Process list of unique indexes to relevant sequences

        :param input: list of unique indexes
        :return: list of frame_id sequences
        """
        pattern = []
        cur_line = []
        for i in input:
            if not cur_line or (i - cur_line[-1]) <= MIN_EMPTY_SEQUENCE:
                cur_line.append(i)
            else:
                pattern.append(cur_line)
                cur_line = [i]

            if i == input[-1]:
                pattern.append(cur_line)
        patten_upd = []
        for i in pattern:
            if len(i) > MIN_OBJ_SEQUENCE:
                patten_upd.append(i)
        return patten_upd

    def fill_new_id(self, frame_id: int, boxes: list) -> dict:
        """
        Fill the new id form with data

        :param frame_id: int frame id
        :param boxes: list of coordinates, ex. [x1, y1, x2, y2]
        """
        bb_dict = {}
        new_id = max(list(self.tracker_dict.keys())) + 1 if self.tracker_dict else 1
        for box in boxes:
            self.add_new_id_form(new_id=new_id)
            self.update_id(
                id=new_id,
                frame_id=frame_id,
                coords=box,
            )
            self.working_idxs.append(new_id)
            bb_dict[new_id] = box
            new_id += 1
        return bb_dict

    # @staticmethod
    def update_id(
            self, id: int, frame_id: int, coords: list,
            # coords_2: list, images,
            relevant: bool = True,
            # status: str = 'run', carpet_size: str = ''
    ):
        """
        Update existing id form with data

        :param id: int tracked id
        :param frame_id: int frame id
        :param coords: list of coordinates, ex. [x1, y1, x2, y2]
        """
        self.tracker_dict[id]['frame_id'].append(frame_id)
        if coords:
            self.tracker_dict[id]['coords'].append(coords)
        # if coords_2:
        #     id_dict[id]['coords_2'].append(coords_2)
        # id_dict[id]['images'].append(images)
        if not relevant:
            self.tracker_dict[id]['relevant'] = relevant
            # id_dict[id]['images'] = []
        if len(self.tracker_dict[id]['coords']) > 1:
            self.tracker_dict[id]['shift'].append(
                self.get_distance(self.tracker_dict[id]['coords'][-2], self.tracker_dict[id]['coords'][-1]))
        # if status == 'end':
        #     id_dict[id]['status'] = status
        #     id_dict[id]['images'] = []
        # if carpet_size:
        #     id_dict[id]['carpet_size'] = carpet_size
        # return id_dict

    def fill_dead_box_form(self, key, frame_id, box):
        """
        Fill the dead box form with data

        :param key: int inner dict id for dead track, id should already be in self.dead_boxes
        :param frame_id: int
        :param box: list of coordinates, ex. [x1, y1, x2, y2]
        """
        self.dead_boxes[key]['frame_id'].append(frame_id)
        self.dead_boxes[key]['coords'].append(box)

    def move_boxes_from_track_to_dead(self, frame_idxs: list, boxes: list):
        max_id = max(list(self.dead_boxes.keys())) if self.dead_boxes else 0
        self.dead_boxes[max_id + 1] = {
            'frame_id': frame_idxs,
            'coords': boxes
        }

    def update_tracker(self, frame_id: int, distance_limit: float):
        """
        Check working ids on old or dead tracks and update tracker dict and working_idxs
        :param frame_id: int
        :param distance_limit: float
        """

        remove_keys = []
        for id in self.working_idxs:

            # check old tracks
            if (frame_id - self.tracker_dict[id]['frame_id'][-1]) > MIN_EMPTY_SEQUENCE:
                remove_keys.append(id)
                if len(self.tracker_dict[id]['coords']) < MIN_OBJ_SEQUENCE:
                    self.tracker_dict[id]['relevant'] = False
                    self.tracker_dict[id]['countable'] = False

            # check dead tracks
            # elif len(self.tracker_dict[id]['shift']) >= MIN_OBJ_SEQUENCE and \
            #         max(self.tracker_dict[id]['shift'][-int(MIN_OBJ_SEQUENCE / 2):]) < distance_limit:
            elif len(self.tracker_dict[id]['shift']) >= MIN_OBJ_SEQUENCE and \
                    max(self.tracker_dict[id]['shift'][-MIN_OBJ_SEQUENCE:]) < distance_limit:
                # self.move_boxes_from_track_to_dead(
                #     frame_idxs=self.tracker_dict[id]['frame_id'][-int(MIN_OBJ_SEQUENCE / 2):],
                #     boxes=self.tracker_dict[id]['coords'][-int(MIN_OBJ_SEQUENCE / 2):]
                # )
                # self.tracker_dict[id]['frame_id'] = self.tracker_dict[id]['frame_id'][:-int(MIN_OBJ_SEQUENCE / 2)]
                # self.tracker_dict[id]['coords'] = self.tracker_dict[id]['coords'][:-int(MIN_OBJ_SEQUENCE / 2)]
                # self.tracker_dict[id]['relevant'] = False
                # remove_keys.append(id)
                self.move_boxes_from_track_to_dead(
                    frame_idxs=self.tracker_dict[id]['frame_id'][-MIN_OBJ_SEQUENCE:],
                    boxes=self.tracker_dict[id]['coords'][-MIN_OBJ_SEQUENCE:]
                )
                self.tracker_dict[id]['frame_id'] = self.tracker_dict[id]['frame_id'][:-MIN_OBJ_SEQUENCE]
                self.tracker_dict[id]['coords'] = self.tracker_dict[id]['coords'][:-MIN_OBJ_SEQUENCE]
                self.tracker_dict[id]['relevant'] = False
                self.tracker_dict[id]['countable'] = False
                remove_keys.append(id)

        self.count = 0
        for k in self.tracker_dict.keys():
            if len(self.tracker_dict[k]['coords']) >= MIN_OBJ_SEQUENCE:
                # print(frame_id, k, len(self.tracker_dict[k]['coords']), self.tracker_dict[k]['countable'],
                #       self.tracker_dict[k]['coords'])
                self.tracker_dict[k]['countable'] = True
            elif frame_id - self.tracker_dict[k]['frame_id'][-1] > MIN_EMPTY_SEQUENCE and MIN_OBJ_SEQUENCE > len(
                    self.tracker_dict[k]['coords']) and k not in remove_keys:
                self.tracker_dict[k]['countable'] = False
                self.tracker_dict[k]['relevant'] = False
                remove_keys.append(k)
            if self.tracker_dict[k]['countable'] and self.tracker_dict[k]['relevant']:
                self.count += 1

        # clear tracker from non-countable and non-relevant boxes
        for k in remove_keys:
            if not self.tracker_dict.get(k).get('countable') and not self.tracker_dict.get(k).get('relevant'):
                self.tracker_dict.pop(k)

        # remove old and dead tracks id from working_idxs
        self.working_idxs = Tracker.lists_difference(self.working_idxs, remove_keys)

    def update_dead_boxes(self, frame_id, new_box, distance_limit):
        """
        Update dead box form with data and remove too old dead tracks

        :param frame_id: int
        :param new_box: list of coordinates, ex. [x1, y1, x2, y2]
        :param distance_limit: float
        :return: initial box, ex. [x1, y1, x2, y2] if not match to any dead tracks
                or empty list if box is in dead track
        """
        drop_keys = []
        find = False
        for k in self.dead_boxes.keys():
            if frame_id - self.dead_boxes.get(k).get('frame_id')[-1] > MIN_EMPTY_SEQUENCE:
                drop_keys.append(k)
            elif self.get_distance(box1=self.dead_boxes.get(k).get('coords')[-1], box2=new_box) < distance_limit:
                self.fill_dead_box_form(key=k, frame_id=frame_id, box=new_box)
                find = True
            else:
                continue
        for dk in drop_keys:
            self.dead_boxes.pop(dk)

        if find:
            return []
        else:
            return new_box

    @staticmethod
    def track_box(boxes, track_dict, working_idxs):
        """
        Find a matching track for boxes and find unmatched boxes and unused tracks ids

        :param boxes: list of boxes, ex. [[x1, y1, x2, y2], ...]
        :param track_dict: self.tra cker_dict
        :param working_idxs: list of actual tracked id, ex. [1, 2, 3]

        :return:    tracked_boxes: dict of tracked id, ex. {1: [x1, y1, x2, y2], 2: [x1, y1, x2, y2]},
                    unused_boxes: list of untracked boxes, ex. [[x1, y1, x2, y2], ...]
                    unused_id_keys: list of id from working_idxs with is not match to boxes, ex. [1, 2, 3]
        """
        boxes_for_track = {}
        for i, b in enumerate(boxes):
            boxes_for_track[i] = b

        cur_boxes = {}
        for i in working_idxs:
            cur_boxes[i] = track_dict[i]['coords'][-1]

        distances = []
        for cur_b in cur_boxes.keys():
            for b_tr in boxes_for_track.keys():
                d = Tracker.get_distance(cur_boxes[cur_b], boxes_for_track[b_tr])
                distances.append((d, (cur_b, b_tr)))

        # unused_id_keys = list(cur_boxes.keys())
        unused_boxes_keys = list(boxes_for_track.keys())

        pairs = []
        tracked_boxes = {}
        while distances:
            closest_pair = sorted(distances)[0]
            pairs.append(closest_pair[1])
            tracked_boxes[closest_pair[1][0]] = boxes_for_track[closest_pair[1][1]]
            # unused_id_keys.pop(unused_id_keys.index(closest_pair[1][0]))
            unused_boxes_keys.pop(unused_boxes_keys.index(closest_pair[1][1]))
            del_keys = []
            for i, key in enumerate(distances):
                if closest_pair[1][0] in key[1] or closest_pair[1][1] in key[1]:
                    del_keys.append(i)
            del_keys = sorted(del_keys, reverse=True)
            for dk in del_keys:
                distances.pop(dk)

        unused_boxes = [boxes_for_track.get(i) for i in unused_boxes_keys]
        return tracked_boxes, unused_boxes

    def sequence_update(self, frame_id, bb1, bb2):

        # if carpet track is started
        if (bb1 or bb2) and self.empty_seq and (frame_id - self.empty_seq[-1] >= MIN_OBJ_SEQUENCE):
            # print(frame_id, self.empty_seq[-1], MIN_OBJ_SEQUENCE)
            self.carpet_seq.append(frame_id)
            self.empty_seq = []
            empty = False
            carpet = True
        elif (bb1 or bb2) and self.empty_seq and (frame_id - self.empty_seq[-1] < MIN_OBJ_SEQUENCE):
            self.carpet_seq.append(frame_id)
            empty = True
            carpet = True
        elif (bb1 or bb2) and not self.empty_seq:
            self.carpet_seq.append(frame_id)
            empty = False
            carpet = True

        # if carpet track is ended
        elif (not bb1 and not bb2) and self.carpet_seq and (frame_id - self.carpet_seq[-1] >= MIN_OBJ_SEQUENCE):
            self.empty_seq.append(frame_id)
            self.carpet_seq = []
            empty = True
            carpet = False
        elif (not bb1 and not bb2) and self.carpet_seq and (frame_id - self.carpet_seq[-1] < MIN_OBJ_SEQUENCE):
            self.empty_seq.append(frame_id)
            empty = True
            carpet = True
        else:
            self.empty_seq.append(frame_id)
            empty = True
            carpet = False
        return empty, carpet

    def process(self, frame_id: int, predict_camera_1: dict, remove_perimeter_boxes: bool = True,
                         speed_limit_percent: float = SPEED_LIMIT_PERCENT):
        bb1 = predict_camera_1.get('boxes')
        # print(bb1)
        origin_shape_1 = predict_camera_1.get('orig_shape')
        speed_limit = speed_limit_percent * (origin_shape_1[0] ** 2 + origin_shape_1[1] ** 2) ** 0.5

        # Remove perimeter irrelevant boxes if condition is True
        if bb1 and remove_perimeter_boxes:
            # bb1 = [Tracker.remove_perimeter_boxes(bb, origin_shape_1) for bb in bb1] \
            #     if remove_perimeter_boxes else bb1
            bb = []
            for box in bb1:
                box = Tracker.remove_perimeter_boxes(box, origin_shape_1)
                if box:
                    bb.append(box)
            bb1 = bb
            # print(f"after removing perimeter boxes: {bb1}")

        # Check in dead boxes list and update it
        if bb1 and self.dead_boxes:
            bb1_upd = []
            for box in bb1:
                b = self.update_dead_boxes(frame_id=frame_id, new_box=box, distance_limit=speed_limit)
                if b:
                    bb1_upd.append(b)
            bb1 = bb1_upd
            # print(f"after removing dead boxes: {bb1}")

        # Track boxes
        if bb1:
            # Track boxes and write in track dict or list of untracked boxes
            tracked_boxes, untracked_boxes = self.track_box(
                boxes=bb1,
                track_dict=self.tracker_dict,
                working_idxs=self.working_idxs
            )
            # print(tracked_boxes, untracked_boxes)
            if tracked_boxes:
                for k, v in tracked_boxes.items():
                    self.update_id(
                        id=k,
                        frame_id=frame_id,
                        coords=v
                    )
            if untracked_boxes:
                new_tracked_boxes = self.fill_new_id(
                    frame_id=frame_id,
                    boxes=untracked_boxes
                )
                tracked_boxes.update(new_tracked_boxes)
            self.current_id = list(tracked_boxes.keys())
            self.current_boxes = list(tracked_boxes.values())

        self.update_tracker(frame_id=frame_id, distance_limit=speed_limit)


def detect_video(model, video_path, save_path, remove_perimeter_boxes=False, start=0, finish=0,
                 speed_limit=SPEED_LIMIT_PERCENT):
    # Get names and colors
    names = ['carpet']
    colors = get_colors(names)

    # video_path = '/home/deny/Рабочий стол/CarpetTracker/videos/Test_0.mp4'
    # save_path = 'tracked_video.mp4'
    tracker = Tracker()

    vc = cv2.VideoCapture()
    vc.open(video_path)
    f = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    finish = int(f) if finish == 0 or finish < start else finish
    fps = vc.get(cv2.CAP_PROP_FPS)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # model = YOLO('runs/detect/train21/weights/best.pt')
    if save_path:
        # print(save_path, fps, (w, h))
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))

    # result = {}
    for i in range(int(f)):
        ret, frame = vc.read()
        if start <= i < finish:
            # print(f"Processed {i + 1} / {f} frames")
            res = model(frame)
            result = {'boxes': res[0].boxes.boxes.tolist(), 'orig_shape': res[0].orig_shape}
            # print(result.get('boxes').boxes.tolist())
            # break
            tracker.process(
                frame_id=i,
                predict_camera_1=result,
                remove_perimeter_boxes=remove_perimeter_boxes,
                speed_limit_percent=speed_limit
            )

            if save_path:
                if tracker.current_id:
                    tracker_id = tracker.current_id
                    coords = tracker.current_boxes
                    labels = [
                        f"# {tracker_id[i]} {model.model.names[coords[i][-1]]} {coords[i][-2]:0.2f}"
                        for i in range(len(tracker_id))
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
                        coordinates=coords
                    )
                    fr = np.array(fr)
                else:
                    fr = res[0].orig_img[:, :, ::-1].copy()

                headline = f"Обнаружено объектов: {tracker.count}\n"
                fr = add_headline_to_cv_image(
                    image=fr,
                    headline=headline
                )
                cv_img = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                out.write(cv_img)
                tracker.current_id = []
                tracker.current_boxes = []
                if i >= finish:
                    break
    print(tracker.tracker_dict)


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
            if i > finish:
                out.release()
                break


if __name__ == "__main__":
    TEST_1_VIDEO_PREDICT = True
    # model1 = YOLO('runs/detect/camera_1_mix+_8n_100ep/weights/best.pt')
    # detect_video(
    #     model=model1,
    #     video_path='videos/sync_test/test 5_cam 1_sync.mp4',
    #     save_path=f'temp/test.mp4',
    #     remove_perimeter_boxes=False,
    #     start=1500,
    #     finish=2000
    # )

    TEST_2_VIDEO_PREDICT = True
    models = {
        'model_1': YOLO('runs/detect/camera_1_mix++_8n_200ep/weights/best.pt'),
        'model_2': YOLO('runs/detect/camera_2_mix++_8n_200ep/weights/best.pt')
    }
    video_paths = [
        {
            'model_1': 'videos/sync_test/test 4_cam 1_sync.mp4',
            'model_2': 'videos/sync_test/test 4_cam 2_sync.mp4',
        },
        {
            'model_1': 'videos/sync_test/test 5_cam 1_sync.mp4',
            'model_2': 'videos/sync_test/test 5_cam 2_sync.mp4',
        },
    ]
    save_path = [
        f'temp/test 4.mp4',
        f'temp/test 5.mp4',
    ]
    for i in range(len(video_paths)):
        detect_synchro_video(
            models=models,
            video_paths=video_paths[i],
            save_path=save_path[i],
            start=0,
            finish=0,
            speed_limit=SPEED_LIMIT_PERCENT
        )
    # for sp in [0.005, 0.01, 0.02, 0.03]:
    #     detect_synchro_video(
    #         models=models,
    #         video_paths=video_paths[0],
    #         save_path=f'temp/test_sp{sp}.mp4',
    #         start=0,
    #         finish=0,
    #         speed_limit=sp
    #     )

    TEST_FUNCS = True
    # def get_pattern(input: list):
    #     max_range = MIN_OBJ_SEQUENCE
    #     pattern = []
    #     cur_line = []
    #     for id in input:
    #         if not cur_line or (id - cur_line[-1]) <= max_range:
    #             cur_line.append(id)
    #         else:
    #             pattern.append(cur_line)
    #             cur_line = [id]
    #
    #         if id == input[-1]:
    #             pattern.append(cur_line)
    #     patten_upd = []
    #     for id in pattern:
    #         if len(id) > max_range:
    #             patten_upd.append(id)
    #     return patten_upd
    #
    #
    # tr1 = Tracker()
    # tr2 = Tracker()
    # # for i in range(len(tracker_1_coord)):
    # origin_shape_1 = (1080, 1920)
    # origin_shape_2 = (360, 640)
    # speed_limit_1 = 0.01 * (origin_shape_1[0] ** 2 + origin_shape_1[1] ** 2) ** 0.5
    # speed_limit_2 = 0.01 * (origin_shape_2[0] ** 2 + origin_shape_2[1] ** 2) ** 0.5
    # # print(f"shift limit ={speed_limit}")
    # for i in range(1500, 2000):
    #     # if len(tracker_1_coord[i]) > 1:
    #     #     print(i, tracker_1_coord[i])
    #     bb1 = {'boxes': tracker_1_coord[i], 'orig_shape': origin_shape_1}
    #     bb2 = {'boxes': tracker_2_coord[i], 'orig_shape': origin_shape_2}
    #     tr1.process_camera_1(
    #         frame_id=i,
    #         predict_camera_1=bb1,
    #         # predict_camera_2=tracker_2_coord[i],
    #     )
    #     tr2.process_camera_1(
    #         frame_id=i,
    #         predict_camera_1=bb2,
    #         # predict_camera_2=tracker_2_coord[i],
    #     )
    #     pattern = []
    #     for k in tr1.tracker_dict.keys():
    #         pattern.extend(tr1.tracker_dict[k]['frame_id'])
    #     for k in tr2.tracker_dict.keys():
    #         pattern.extend(tr2.tracker_dict[k]['frame_id'])
    #     pattern = list(set(pattern))
    #     pattern = get_pattern(pattern)
    #
    #     logger.info(
    #         f"{i}, cur_box={tracker_1_coord[i]}, count={tr1.count}, working_idxs={tr1.working_idxs}, "
    #         f"dead count={len(tr1.dead_boxes.keys())}, all idxs={list(tr1.tracker_dict.keys())}")
    #     logger.info(
    #         f"{i}, cur_box={tracker_2_coord[i]}, count={tr2.count}, working_idxs={tr2.working_idxs}, "
    #         f"dead count={len(tr2.dead_boxes.keys())}, all idxs={list(tr2.tracker_dict.keys())}")
    #     logger.info(
    #         f"{i}, pattern count={len(pattern)}, pattern={pattern}\n")
    #     # if i == 1000:
    #     #     # es = tr.empty_seq[-MIN_OBJ_SEQUENCE:] if len(tr.empty_seq) >= MIN_OBJ_SEQUENCE else tr.empty_seq
    #     #     # cs = tr.carpet_seq[-MIN_OBJ_SEQUENCE:] if len(tr.carpet_seq) >= MIN_OBJ_SEQUENCE else tr.carpet_seq
    #     #     # print(i, 'empty_seq=', es, 'carpet_seq=', cs, 'predict=', tracker_1_coord[i], tracker_2_coord[i])
    #     #     # print(i, 'predict=', tracker_1_coord[i], tracker_2_coord[i])
    #     #     break
    #     #     # print(i, tracker_1_coord[i], tr.count, tr.working_idxs, tr.tracker_dict.keys())
    # print(tr1.count, tr1.tracker_dict)
    # print(tr2.count, tr2.tracker_dict)
    # print(len(pattern), pattern)
