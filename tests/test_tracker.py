import copy
import os
from collections import Counter

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.path as mpltPath
from seaborn._stats.counting import Count
from torch import nn
from torchvision.utils import draw_bounding_boxes

from nn_classificator import VideoClassifier
from parameters import *
from utils import load_data, time_converter, get_colors, add_headline_to_cv_image, logger

# POLY_CAM1_IN = [[110, 0], [410, 650], [725, 415], [545, 0]]
# POLY_CAM1_OUT = [[0, 0], [330, 690], [540, 705], [810, 630]([775, 525]), [870, 465], [855, 315], [760, 0]]
# POLY_CAM2_IN = [[140, 0], [140, 150], [260, 185], [260, 0]]
# POLY_CAM2_OUT = [[100, 0], [100, 140], [105, 195], [160, 255], [250, 250], [310, 200], [310, 0]]

# POLY_CAM1_IN = [[0.0573, 0.0], [0.2135, 0.6019], [0.3776, 0.3843], [0.2839, 0.0]]
POLY_CAM1_OUT = [[0.0, 0.0], [0.0, 0.4167], [0.1718, 0.6389], [0.2813, 0.7083], [0.3984, 0.5833], [0.4218, 0.4306],
                 [0.4141, 0.0]]
# POLY_CAM2_IN = [[0.2187, 0.0], [0.2187, 0.4167], [0.4062, 0.5139], [0.4062, 0.0]]
POLY_CAM2_OUT = [[0.0938, 0.0], [0.1406, 0.5], [0.25, 0.6667],
                 [0.3906, 0.6944], [0.5156, 0.5277], [0.6718, 0.0]]

GLOBAL_STEP = 0.1


class PolyTracker:
    def __init__(self,
                 polygon_in: list,
                 polygon_out: list,
                 name: str = ''):
        self.count_frames = []
        self.name = name
        self.frame_id = 0
        self.track_list = []
        self.count = 0
        self.max_id = 0
        self.current_boxes = []
        self.polygon_in = polygon_in
        self.polygon_out = polygon_out
        self.dead_boxes = {}

    @staticmethod
    def prepare_image(image, colors, tracker_current_boxes, poly_width, reshape,  # polygon_in,
                      polygon_out):
        # Draw all figures on image
        # img = PolyTracker.draw_polygons(polygons=polygon_in, image=image, outline=(0, 255, 0), width=poly_width)
        img = PolyTracker.draw_polygons(polygons=polygon_out, image=image, outline=(0, 0, 255), width=poly_width)

        labels = [f"carpet" for _ in tracker_current_boxes]
        current_boxes = [tr[0] for tr in tracker_current_boxes]
        if len(labels) > 1:
            cl = colors * len(labels)
        else:
            cl = colors

        img = PolyTracker.put_box_on_image(
            save_path=None,
            image=img,
            labels=labels,
            color_list=cl,
            coordinates=current_boxes,
        )
        for box in current_boxes:
            img = cv2.circle(img, PolyTracker.get_center([int(c) for c in box[:4]]), int(0.01 * img.shape[0]),
                             colors[0], -1)

        img = np.array(img)
        img = cv2.resize(img, reshape)
        return img

    @staticmethod
    def draw_polygons(polygons: list, image: np.ndarray, outline=(0, 200, 0), width: int = 5) -> np.ndarray:
        if type(polygons[0]) == list:
            xy = []
            for i in polygons:
                xy.append(i[0])
                xy.append(i[1])
        else:
            xy = polygons

        points = np.array(xy)
        points = points.reshape((-1, 1, 2))
        image = cv2.polylines(image, [points], True, outline, width)
        return np.array(image)

    @staticmethod
    def point_in_polygon(point: list, polygon: list[list, ...]) -> bool:
        path = mpltPath.Path(polygon)
        return path.contains_points([point])[0]

    @staticmethod
    def get_center(coord: list[int, int, int, int]) -> list[int, int]:
        return [int((coord[0] + coord[2]) / 2), int((coord[1] + coord[3]) / 2)]

    @staticmethod
    def get_distance(box1: list, box2: list) -> float:
        """
        :param box1: list of coordinates with first 4 index positions [x1, y1, x2, y2]
        :param box2: list of coordinates with first 4 index positions [x1, y1, x2, y2]
        :return: float distance between the two box centers
        """
        c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    @staticmethod
    def put_box_on_image(save_path, image, labels, color_list, coordinates):
        image = np.transpose(image, (2, 0, 1))
        w, h = image.shape[:2]
        image = torch.from_numpy(image)
        coord = []
        for box in coordinates:
            coord.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        bbox = torch.tensor(coord, dtype=torch.int)
        if bbox.tolist():
            image_true = draw_bounding_boxes(
                image, bbox, width=3, labels=labels, colors=color_list, fill=True,
                font=os.path.join(ROOT_DIR, "arial.ttf"), font_size=int(h * 0.02))
            image = torchvision.transforms.ToPILImage()(image_true)
        else:
            image = torchvision.transforms.ToPILImage()(image)
        if save_path:
            image.save(f'{save_path}')
        return np.array(image)

    @staticmethod
    def add_track():
        return dict(id=None, boxes=[], frame_id=[], check_in=[], check_out=[], shift_center=[], speed=[],
                    shift_top_left=[], shift_top_right=[], shift_bottom_left=[], shift_bottom_right=[])

    @staticmethod
    def fill_track(track: dict, id: int, frame_id: int, box: list, check_in: bool, check_out: bool):
        track['id'] = id
        track['frame_id'].append(frame_id)
        track['check_in'].append(check_in)
        track['check_out'].append(check_out)
        if track['boxes']:
            shift_center, shift_top_left, shift_top_right, shift_bottom_left, shift_bottom_right = \
                PolyTracker.get_full_distance(track['boxes'][-1], box)
            track['shift_center'].append(shift_center)
            track['shift_top_left'].append(shift_top_left)
            track['shift_top_right'].append(shift_top_right)
            track['shift_bottom_left'].append(shift_bottom_left)
            track['shift_bottom_right'].append(shift_bottom_right)
            track['speed'].append(
                abs(PolyTracker.get_distance(box1=box, box2=track['boxes'][-1])) / (frame_id - track['frame_id'][-2]))
        track['boxes'].append(box)
        return track

    @staticmethod
    def expand_poly(poly, step):
        def poly_center_coord(poly):
            x = np.mean([i[0] for i in poly])
            y = np.mean([i[1] for i in poly])
            return int(x), int(y)

        center = poly_center_coord(poly)
        exp_poly = []
        for coord in poly:
            distance = ((coord[0] - center[0]) ** 2 + (coord[1] - center[1]) ** 2) ** 0.5
            new_distance = distance + step
            new_x = new_distance / distance * (coord[0] - center[0]) + center[0]
            new_y = new_distance / distance * (coord[1] - center[1]) + center[1]
            exp_poly.append([int(new_x), int(new_y)])
        return exp_poly

    def update_dead_boxes(self, frame_id, new_box, distance_limit):
        """
        Update dead box form with data and remove too old dead tracks

        :param frame_id: int
        :param new_box: list of coordinates, ex. [x1, y1, x2, y2]
        :param distance_limit: float
        :return: initial box, ex. [x1, y1, x2, y2] if not match to any dead tracks
                or empty list if box is in dead track
        """
        # new_box = [int(x) for x in new_box[:4]]
        drop_keys = []
        find = False
        for k in self.dead_boxes.keys():
            shift_center, shift_top_left, shift_top_right, shift_bottom_left, shift_bottom_right = \
                self.get_full_distance(box1=self.dead_boxes.get(k).get('coords')[-1], box2=new_box)
            if frame_id - self.dead_boxes.get(k).get('frame_id')[-1] > MIN_EMPTY_SEQUENCE:
                drop_keys.append(k)
            elif shift_center < distance_limit or shift_top_left < distance_limit or \
                    shift_top_right < distance_limit or shift_bottom_left < distance_limit or \
                    shift_bottom_right < distance_limit:
                self.fill_dead_box_form(key=k, frame_id=frame_id, box=new_box)
                find = True
            else:
                continue
        for dk in drop_keys:
            self.dead_boxes.pop(dk)

        if find:
            return []
        return new_box

    def fill_dead_box_form(self, key, frame_id, box):
        """
        Fill the dead box form with data

        :param key: int inner dict id for dead track, id should already be in self.dead_boxes
        :param frame_id: int
        :param box: list of coordinates, ex. [x1, y1, x2, y2]
        """
        # box = [int(x) for x in box[:4]]
        self.dead_boxes[key]['frame_id'].append(frame_id)
        self.dead_boxes[key]['coords'].append(box)

    @staticmethod
    def get_full_distance(box1: list, box2: list) -> tuple[float, float, float, float, float]:
        """
        :param box1: list of coordinates with first 4 index positions [x1, y1, x2, y2]
        :param box2: list of coordinates with first 4 index positions [x1, y1, x2, y2]
        :return: float distance between the two box centers
        """
        c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        shift_center = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        shift_top_left = ((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2) ** 0.5
        shift_top_right = ((box1[0] - box2[0]) ** 2 + (box1[3] - box2[3]) ** 2) ** 0.5
        shift_bottom_left = ((box1[2] - box2[2]) ** 2 + (box1[1] - box2[1]) ** 2) ** 0.5
        shift_bottom_right = ((box1[2] - box2[2]) ** 2 + (box1[3] - box2[3]) ** 2) ** 0.5
        return shift_center, shift_top_left, shift_top_right, shift_bottom_left, shift_bottom_right

    def move_boxes_from_track_to_dead(self, frame_idxs: list, boxes: list):
        max_id = max(list(self.dead_boxes.keys())) if self.dead_boxes else 0
        self.dead_boxes[max_id + 1] = {
            'frame_id': frame_idxs,
            'coords': boxes
        }

    @staticmethod
    def predict_track_class(model: nn.Module, tracks, classes, frame_size: tuple[int, int]):
        # vc = VideoClassifier(num_classes=len(classes), weights=model_weights)
        arr = VideoClassifier.track_to_array(
            tracks=[tracks['tr1'], tracks['tr2']], frame_size=frame_size, num_frames=model.input_size[0]
        )
        return vc.predict(arr, model=model, classes=classes)

    @staticmethod
    def combine_count(frame_id: int, count: int, last_track_seq: dict, class_counter: list, class_list: list,
                      tracker_1_count_frames: list, tracker_2_count_frames: list, class_model: nn.Module,
                      existing_tracks: list[int, int], frame_size: tuple[int, int],
                      debug: bool = False, stop_flag: bool = False, ) -> (list, dict):
        print(f"combine_count: frame_id={frame_id}, count={count}, last_track_seq={last_track_seq}, "
              f"tracker_1_count_frames={tracker_1_count_frames}, tracker_2_count_frames={tracker_2_count_frames}")

        last_state = [False, False]
        for p, key in enumerate(last_track_seq.keys()):
            if not last_track_seq[key]:
                last_state[p] = False
            else:
                last_state[p] = True

        new_track_seq = {'tr1': [], 'tr2': []}
        new_state = [False, False]
        if tracker_1_count_frames:
            new_track_seq['tr1'] = tracker_1_count_frames
            new_state[0] = True

        if tracker_2_count_frames:
            new_track_seq['tr2'] = tracker_2_count_frames
            new_state[1] = True

        if new_state != [False, False] and debug:
            print(f"-- frame_id={frame_id}, count {count}, last_state={last_state}, new_state={new_state}, "
                  f"last_track_seq={last_track_seq}, new_track_seq={new_track_seq}")

        limit = MIN_EMPTY_SEQUENCE
        if new_state == [False, False] and last_state == [False, False]:
            return class_counter, last_track_seq

        elif new_state == [False, False] and last_state != [False, False]:
            max_last_1 = max(last_track_seq['tr1'][0]) if last_track_seq['tr1'] else 0
            max_last_2 = max(last_track_seq['tr2'][0]) if last_track_seq['tr2'] else 0
            if (last_state == [True, True] and frame_id - max([max_last_1, max_last_2]) > limit) or \
                    (last_state != [False, False] and last_state != [True, True] and existing_tracks == [0, 0]):
                predict_class = PolyTracker.predict_track_class(
                    model=class_model, tracks=last_track_seq, classes=class_list, frame_size=frame_size)
                class_counter.append(predict_class[0])
                print(1, "frame_id", frame_id, "last_track_seq", last_track_seq)
                last_track_seq['tr1'] = []
                last_track_seq['tr2'] = []
                return class_counter, last_track_seq
            return class_counter, last_track_seq

        elif new_state != [False, False] and last_state == [False, False]:
            if new_state[0]:
                last_track_seq['tr1'] = new_track_seq['tr1']
            if new_state[1]:
                last_track_seq['tr2'] = new_track_seq['tr2']
            return class_counter, last_track_seq

        elif last_state == [True, True] or new_state == [True, True] or \
                (last_state == [True, False] and new_state == [True, False]) or \
                (last_state == [False, True] and new_state == [False, True]):
            predict_class = PolyTracker.predict_track_class(
                model=class_model, tracks=last_track_seq, classes=class_list, frame_size=frame_size)
            class_counter.append(predict_class[0])
            print(2, "frame_id", frame_id, "last_track_seq", last_track_seq)
            last_track_seq['tr1'] = new_track_seq['tr1'] if new_state[0] else []
            last_track_seq['tr2'] = new_track_seq['tr2'] if new_state[1] else []
            return class_counter, last_track_seq

        else:
            last_track_seq['tr1'] = new_track_seq['tr1'] if new_state[0] else last_track_seq['tr1']
            last_track_seq['tr2'] = new_track_seq['tr2'] if new_state[1] else last_track_seq['tr2']
            return class_counter, last_track_seq

        # if new_state == [False, False]:
        #     pass
        #
        # elif last_state == [False, False] and new_state == [True, False]:
        #     if debug:
        #         print(1)
        #     if existing_tracks == [0, 0]:
        #         predict_class = PolyTracker.predict_track_class(
        #             model=model, tracks=last_track_seq, classes=class_list)
        #         class_counter[predict_class] += 1
        #         count += 1
        #     last_track_seq['tr1'] = new_track_seq['tr1']
        #     # count += 1
        #
        # elif last_state == [True, False] and new_state == [True, False]:
        #     if debug:
        #         print(1-1)
        #     predict_class = PolyTracker.predict_track_class(
        #         model=model, tracks=last_track_seq, classes=class_list)
        #     class_counter[predict_class] += 1
        #     last_track_seq['tr1'] = new_track_seq['tr1']
        #     count += 1
        #
        # elif last_state == [False, False] and new_state == [False, True]:
        #     if debug:
        #         print(2)
        #     if existing_tracks == [0, 0]:
        #         predict_class = PolyTracker.predict_track_class(
        #             model=model, tracks=last_track_seq, classes=class_list)
        #         class_counter[predict_class] += 1
        #         count += 1
        #     last_track_seq['tr2'] = new_track_seq['tr2']
        #     # count += 1
        #
        # elif last_state == [False, True] and new_state == [False, True]:
        #     if debug:
        #         print(2)
        #     last_track_seq['tr2'] = new_track_seq['tr2']
        #     count += 1
        #
        # elif last_state == [True, False] and new_state == [False, True]:
        #     if min(new_track_seq['tr2'][0]) - max(last_track_seq['tr1'][0]) > limit:
        #         if debug:
        #             print(3)
        #         last_track_seq['tr1'] = []
        #         last_track_seq['tr2'] = new_track_seq['tr2']
        #         count += 1
        #     else:
        #         if debug:
        #             print(4)
        #         last_track_seq['tr2'] = new_track_seq['tr2']
        #
        # elif last_state == [False, True] and new_state == [True, False]:
        #     if min(new_track_seq['tr1'][0]) - max(last_track_seq['tr2'][0]) > limit:
        #         if debug:
        #             print(5)
        #         last_track_seq['tr2'] = []
        #         last_track_seq['tr1'] = new_track_seq['tr1']
        #         count += 1
        #     else:
        #         if debug:
        #             print(6)
        #         last_track_seq['tr1'] = new_track_seq['tr1']
        #
        # elif last_state == [True, True] and new_state == [True, False]:
        #     if debug:
        #         print(7)
        #     last_track_seq['tr2'] = []
        #     last_track_seq['tr1'] = new_track_seq['tr1']
        #     count += 1
        #
        # elif last_state == [True, True] and new_state == [False, True]:
        #     if debug:
        #         print(9)
        #     last_track_seq['tr1'] = []
        #     last_track_seq['tr2'] = new_track_seq['tr2']
        #     count += 1
        #
        # else:
        #     if debug:
        #         print(12)
        #     last_track_seq['tr1'] = new_track_seq['tr1']
        #     last_track_seq['tr2'] = new_track_seq['tr2']
        #     count += 1

        # return class_counter, last_track_seq

    @staticmethod
    def isin(pattern, sequence):
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i + len(pattern)] == pattern:
                return True
        return False

    def check_closest_track(self, check_id, speed_limit):
        check_track = self.track_list[check_id]
        print("check_track", check_track, check_id)
        last_fr = check_track['frame_id'][-1]
        closest_id = []
        for i, track in enumerate(self.track_list):
            intersection = list(set(track['frame_id']) & set(check_track['frame_id']))
            if track != check_track and min(track['frame_id']) - max(check_track['frame_id']) < MIN_EMPTY_SEQUENCE \
                    and not len(intersection):
                fr_id = 0
                for fr in track['frame_id']:
                    if fr > last_fr:
                        fr_id = track['frame_id'].index(fr)
                        break
                distance = self.get_distance(check_track['boxes'][-1], track['boxes'][fr_id]) / (
                        track['frame_id'][fr_id] - last_fr)
                closest_id.append((distance, i, fr, last_fr))
            elif track != check_track and min(track['frame_id']) - max(check_track['frame_id']) < MIN_EMPTY_SEQUENCE \
                    and 0 < len(intersection) < 3:
                min_frame = min(intersection)
                fr_id = 0
                for fr in check_track['frame_id']:
                    if fr >= min_frame:
                        break
                    fr_id = check_track['frame_id'].index(fr)
                distance = self.get_distance(check_track['boxes'][fr_id],
                                             track['boxes'][track['boxes'].index(min_frame)])
                closest_id.append((distance, i))
        id = None
        if closest_id:
            closest_id = sorted(closest_id)
            # if closest_id[0][0] < speed_limit:
            id = closest_id[0][1]
        # print('closest_id', closest_id, 'id', id)
        # intersection = list(set(track_list[closest_id]['frames']) & set(check_track['frames']))
        if id is not None:
            min_frame = min([min(check_track['frame_id']), min(self.track_list[id]['frame_id'])])
            max_frame = max([max(check_track['frame_id']), max(self.track_list[id]['frame_id'])])
            new_track = self.add_track()
            # new_track['id'] = track_list[closest_id]['id']
            for fr in range(min_frame, max_frame):
                if fr in self.track_list[id]['frame_id'] and fr in check_track['frame_id']:
                    # print(1)
                    ch_fr_id = check_track['frame_id'].index(fr) - 1 if check_track['frame_id'].index(fr) > 0 else 0
                    fr_id = self.track_list[id]['frame_id'].index(fr) if check_track['frame_id'].index(fr) > 0 \
                        else self.track_list[id]['frame_id'].index(fr) + 1
                    dist1 = self.get_distance(check_track['boxes'][ch_fr_id], self.track_list[id]['boxes'][fr_id])
                    dist2 = self.get_distance(
                        self.track_list[id]['boxes'][fr_id - 1], self.track_list[id]['boxes'][fr_id])
                    if dist1 > dist2:
                        new_track = self.fill_track(
                            track=new_track,
                            id=self.track_list[id]['id'],
                            frame_id=fr,
                            box=self.track_list[id]['boxes'][self.track_list[id]['frame_id'].index(fr)],
                            # check_in=self.track_list[id]['check_in'][self.track_list[id]['frame_id'].index(fr)],
                            check_out=self.track_list[id]['check_out'][self.track_list[id]['frame_id'].index(fr)]
                        )
                    else:
                        new_track = self.fill_track(
                            track=new_track,
                            id=self.track_list[id]['id'],
                            frame_id=fr,
                            box=check_track['boxes'][check_track['frame_id'].index(fr)],
                            check_out=check_track['check_out'][check_track['frame_id'].index(fr)],
                            # check_in=check_track['check_in'][check_track['frame_id'].index(fr)],
                        )
                elif fr in self.track_list[id]['frame_id']:
                    # print(2)
                    new_track = self.fill_track(
                        track=new_track,
                        id=self.track_list[id]['id'],
                        frame_id=fr,
                        box=self.track_list[id]['boxes'][self.track_list[id]['frame_id'].index(fr)],
                        # check_in=self.track_list[id]['check_in'][self.track_list[id]['frame_id'].index(fr)],
                        check_out=self.track_list[id]['check_out'][self.track_list[id]['frame_id'].index(fr)]
                    )
                elif fr in check_track['frame_id']:
                    # print(3)
                    new_track = self.fill_track(
                        track=new_track,
                        id=self.track_list[id]['id'],
                        frame_id=fr,
                        box=check_track['boxes'][check_track['frame_id'].index(fr)],
                        # check_in=check_track['check_in'][check_track['frame_id'].index(fr)],
                        check_out=check_track['check_out'][check_track['frame_id'].index(fr)]
                    )

            print("new_track", new_track)
            self.track_list[id] = new_track
            self.track_list.pop(check_id)
        # return track_list

    def update_track_list(self, distance_limit: float, img_shape: tuple, stop_flag: bool = False, debug: bool = False):
        rel, deleted = [], []
        self.count_frames = []
        cut_idxs = []
        cut = MIN_EMPTY_SEQUENCE
        for i, trck in enumerate(self.track_list):
            if len(trck['shift_center']) > MIN_EMPTY_SEQUENCE and \
                    (
                            max(trck['shift_center'][-cut:]) < distance_limit or
                            max(trck['shift_top_left'][-cut:]) < distance_limit or
                            max(trck['shift_bottom_right'][-cut:]) < distance_limit or
                            max(trck['shift_top_right'][-cut:]) < distance_limit or
                            max(trck['shift_bottom_left'][-cut:]) < distance_limit
                    ):
                self.move_boxes_from_track_to_dead(
                    frame_idxs=trck['frame_id'][-cut:],
                    boxes=trck['boxes'][-cut:]
                )
                trck['frame_id'] = trck['frame_id'][:-cut]
                trck['boxes'] = trck['boxes'][:-cut]
                trck['check_in'] = trck['check_in'][:-cut]
                trck['check_out'] = trck['check_out'][:-cut]
                trck['shift_center'] = trck['shift_center'][:-cut]
                trck['shift_top_left'] = trck['shift_top_left'][:-cut]
                trck['shift_bottom_right'] = trck['shift_bottom_right'][:-cut]
                trck['shift_top_right'] = trck['shift_top_right'][:-cut]
                trck['shift_bottom_left'] = trck['shift_bottom_left'][:-cut]
                cut_idxs.append(i)

        if cut_idxs and self.track_list:
            cut_idxs = sorted(cut_idxs, reverse=True)
            for cut_id in cut_idxs:
                if not self.track_list[cut_id]['boxes']:
                    self.track_list.pop(cut_id)

        for i, trck in enumerate(self.track_list):
            if debug:
                print('trck', trck)

            if stop_flag:
                deleted.append(i)
                self.count += 1

            elif self.frame_id - trck['frame_id'][-1] > MIN_EMPTY_SEQUENCE and \
                    trck['check_in'][-1]:
                continue

            elif self.frame_id - trck['frame_id'][-1] > MIN_EMPTY_SEQUENCE and \
                    trck['check_out'][-1] and not self.isin([False, False, False], trck['check_out']):
                continue

            elif (self.frame_id - trck['frame_id'][-1]) > MIN_EMPTY_SEQUENCE and not trck['check_out'][-1]:
                if debug:
                    print(f"Check {self.name}", -(MIN_EMPTY_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1),
                          sum(trck['check_out'][:-(MIN_EMPTY_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]))
                deleted.append(i)
                self.count += 1

            elif (self.frame_id - trck['frame_id'][-1]) > MIN_EMPTY_SEQUENCE and \
                    self.isin([False, False, False], trck['check_out']):
                if debug:
                    print(f"Check {self.name}", -(MIN_EMPTY_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1),
                          sum(trck['check_out'][:-(MIN_EMPTY_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]))
                deleted.append(i)
                self.count += 1

            else:
                rel.append(i)

        # for i in deleted:
        #     self.count_frames.extend(self.track_list[i]['frame_id'])
        # self.count_frames = list(set(self.count_frames))
        # self.track_list = [self.track_list[i] for i in rel]

        max_len = 0
        for i in deleted:
            if max_len < len(self.track_list[i]['frame_id']):
                self.count_frames = [
                    self.track_list[i]['frame_id'],
                    [[b / img_shape[i % 2] for i, b in enumerate(bb)] for bb in self.track_list[i]['boxes']]
                ]
        # self.count_frames = list(set(self.count_frames))
        self.track_list = [self.track_list[i] for i in rel]

    def process(self, frame_id: int, boxes: list, img_shape: tuple, speed_limit_percent: float = SPEED_LIMIT_PERCENT,
                stop_flag: bool = False, debug: bool = False):
        # check if boxes are relevant
        self.current_boxes = []
        self.frame_id = frame_id
        diagonal = ((img_shape[0]) ** 2 + (img_shape[1]) ** 2) ** 0.5
        speed_limit = speed_limit_percent * diagonal * 0.8
        dist_limit = DEAD_LIMIT_PERCENT * diagonal
        limit_in = self.expand_poly(self.polygon_in, -1 * diagonal * 0.0)
        limit_out = self.expand_poly(self.polygon_out, diagonal * 0.3)
        if debug:
            print("================================")
            print('boxes', boxes)
        for box in boxes:
            box = [int(x) for x in box[:4]]
            center = self.get_center(box)
            # print(self.point_in_polygon(center, limit_in), self.point_in_polygon(center, limit_out))
            # if not self.point_in_polygon(center, limit_in) and self.point_in_polygon(center, limit_out):
            if self.point_in_polygon(center, limit_out):  # and not self.point_in_polygon(center, limit_in):
                check_in = self.point_in_polygon(center, limit_in)
                check_out = self.point_in_polygon(center, self.polygon_out)

                # Check in dead boxes list and update it
                if self.dead_boxes:
                    box = self.update_dead_boxes(frame_id=frame_id, new_box=box, distance_limit=dist_limit)

                if box:
                    self.current_boxes.append([box, check_in, check_out])
        # print("self.current_boxes", self.current_boxes)
        # If no track in list - write new track
        if not self.track_list:
            for box in self.current_boxes:
                if box[-1]:
                    track = self.add_track()
                    track = self.fill_track(
                        track=track,
                        id=self.max_id + 1,
                        frame_id=frame_id,
                        box=[int(c) for c in box[0][:4]],
                        check_in=box[1],
                        check_out=box[2]
                    )

                    self.max_id += 1
                    self.track_list.append(track)
                else:
                    self.move_boxes_from_track_to_dead(frame_idxs=[frame_id], boxes=[box[0]])
            # print("if not self.track_list", self.track_list)
        # if track exist - update track
        else:
            tr_idxs = list(range(len(self.track_list)))
            box_idxs = list(range(len(self.current_boxes)))
            dist = []
            pair = []
            for i in tr_idxs:
                for b in box_idxs:
                    c1 = self.get_center(self.track_list[i]['boxes'][-1])
                    c2 = self.get_center(self.current_boxes[b][0])
                    distance = self.get_distance(self.track_list[i]['boxes'][-1], self.current_boxes[b][0])
                    module_x1 = (c2[0] - c1[0]) / abs(c2[0] - c1[0]) if c2[0] - c1[0] else 0
                    module_y1 = (c2[1] - c1[1]) / abs(c2[1] - c1[1]) if c2[1] - c1[1] else 0
                    vector1 = ((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2) ** 0.5
                    if len(self.track_list[i]['boxes']) > 1:
                        c3 = self.get_center(self.track_list[i]['boxes'][-2])
                        module_x2 = (c1[0] - c3[0]) / abs(c1[0] - c3[0]) if c1[0] - c3[0] else 0
                        module_y2 = (c1[1] - c3[1]) / abs(c1[1] - c3[1]) if c1[1] - c3[1] else 0
                        vector2 = ((c1[0] - c3[0]) ** 2 + (c1[1] - c3[1]) ** 2) ** 0.5
                    else:
                        module_x2, module_y2, vector2 = 0, 0, 0
                    vector = ((vector1, module_x1, module_y1), (vector2, module_x2, module_y2))
                    if (i, b) not in pair or (b, i) not in pair:
                        dist.append((distance, i, b, vector))
                        pair.append((i, b))
            dist = sorted(dist)
            if debug:
                print('current boxes', self.current_boxes)
                print('dist =', dist, 'speed_limit =', speed_limit, 'dist_limit =', dist_limit)
            for d in dist:
                if not self.track_list[d[1]]['check_out'][-1] and self.current_boxes[d[2]][1]:
                    continue
                elif tr_idxs and d[1] in tr_idxs and d[2] in box_idxs:
                    if (d[3][0][1] * d[3][1][1] > 0 and d[3][0][2] * d[3][1][2] > 0) or \
                            ((d[3][0][1] * d[3][1][1] <= 0 or d[3][0][2] * d[3][1][2] <= 0) and
                             d[3][0][0] + d[3][1][0] < speed_limit):
                        self.track_list[d[1]] = self.fill_track(
                            track=self.track_list[d[1]],
                            id=self.track_list[d[1]]['id'],
                            frame_id=frame_id,
                            box=[int(c) for c in self.current_boxes[d[2]][0][:4]],
                            check_in=self.current_boxes[d[2]][1],
                            check_out=self.current_boxes[d[2]][2]
                        )
                        tr_idxs.pop(tr_idxs.index(d[1]))
                        box_idxs.pop(box_idxs.index(d[2]))
            if box_idxs:
                for b in box_idxs:
                    # add track if its first point is inside out-polygon
                    if self.current_boxes[b][2]:
                        track = self.add_track()
                        track = self.fill_track(
                            track=track,
                            id=self.max_id + 1,
                            frame_id=frame_id,
                            box=[int(c) for c in self.current_boxes[b][0][:4]],
                            check_in=self.current_boxes[b][1],
                            check_out=self.current_boxes[b][2]
                        )
                        self.max_id += 1
                        self.track_list.append(track)
                    else:
                        key = 0 if not list(self.dead_boxes.keys()) else max(list(self.dead_boxes.keys())) + 1
                        self.dead_boxes[key] = {
                            'frame_id': [],
                            'coords': []
                        }
                        self.fill_dead_box_form(key=key, frame_id=frame_id, box=self.current_boxes[b][0])
        #         print("track", frame_id, [b['boxes'] for b in self.track_list])
        # if debug:
        #     print("track x", self.track_list)

        # Review tracks
        # if debug:
        #     print(
        #         f"track {self.name}, frame {frame_id}, count={self.count}\n"
        #         f"{[i['frame_id'] for i in self.track_list]}\n"
        #         f"{[i['boxes'] for i in self.track_list]}\n"
        #         f"{[i['check_out'] for i in self.track_list]}"
        #     )
        self.update_track_list(distance_limit=dist_limit, img_shape=img_shape, debug=debug, stop_flag=stop_flag)
        if debug:
            # print("track x2", self.count, [i['boxes'] for i in self.track_list], [i['check_out'] for i in self.track_list])
            print('self.dead_boxes', self.dead_boxes)


if __name__ == '__main__':

    # Problem test 21
    # for vvv in [8, 11, 17, 29, 30]:
    test_vid = 15
    model_id = 'mix4+ 350ep'
    # model_id = 'mix+++ 200ep'

    # vid_1 = f'videos/sync_test/test {test_vid}_cam 1_sync.mp4'
    # vid_2 = f'videos/sync_test/test {test_vid}_cam 2_sync.mp4'

    true_bb_1 = load_data(
        pickle_path=os.path.join(ROOT_DIR, f'tests/boxes/true_bb_1_test {test_vid} ({model_id}).dict'))
    print('true_bb_1', len(true_bb_1))
    true_bb_2 = load_data(
        pickle_path=os.path.join(ROOT_DIR, f'tests/boxes/true_bb_2_test {test_vid} ({model_id}).dict'))
    print('true_bb_2', len(true_bb_2))
    start, finish = (0 * 60 + 0) * 25, (0 * 60 + 10) * 25
    # start, finish = 0, min([len(true_bb_1), len(true_bb_2)])
    classes = ['115x200', '115x400', '150x300', '60x90', '85x150']
    class_counter = []  # {cl: 0 for cl in classes}

    names = ['carpet']
    colors = get_colors(names)
    out_size = (640, 360)
    out = cv2.VideoWriter(os.path.join(ROOT_DIR, f'temp/test {test_vid}.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 25,
                          (out_size[0], out_size[1] * 2))
    vc1 = cv2.VideoCapture()
    vc1.open(os.path.join(DATASET_DIR, vid_1))
    f1 = vc1.get(cv2.CAP_PROP_FRAME_COUNT)
    fps1 = vc1.get(cv2.CAP_PROP_FPS)
    w1 = int(vc1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(vc1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vc2 = cv2.VideoCapture()
    vc2.open(os.path.join(DATASET_DIR, vid_2))
    f2 = vc2.get(cv2.CAP_PROP_FRAME_COUNT)
    fps2 = vc2.get(cv2.CAP_PROP_FPS)
    w2 = int(vc2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(vc2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('fps 1 =', vc1.get(cv2.CAP_PROP_FPS), '\nfps 2 =', vc2.get(cv2.CAP_PROP_FPS))

    step = min([fps1, fps2])
    range_1 = [(i, round(i * 1000 / fps1, 1)) for i in range(int(f1))]
    range_2 = [(i, round(i * 1000 / fps2, 1)) for i in range(int(f2))]
    (min_range, max_range) = (range_1, range_2) if step == fps1 else (range_2, range_1)
    (min_vc, max_vc) = (vc1, vc2) if step == fps1 else (vc2, vc1)

    polygon_in_1 = [[int(p[0] * w1), int(p[1] * h1)] for p in POLY_CAM1_IN]
    polygon_out_1 = [[int(p[0] * w1), int(p[1] * h1)] for p in POLY_CAM1_OUT]
    tracker_1 = PolyTracker(polygon_in=polygon_in_1, polygon_out=polygon_out_1, name='camera 1')
    polygon_in_2 = [[int(p[0] * w2), int(p[1] * h2)] for p in POLY_CAM2_IN]
    polygon_out_2 = [[int(p[0] * w2), int(p[1] * h2)] for p in POLY_CAM2_OUT]
    tracker_2 = PolyTracker(polygon_in=polygon_in_2, polygon_out=polygon_out_2, name='camera 2')

    vc = VideoClassifier(
        num_classes=len(classes),
        weights=os.path.join(ROOT_DIR, 'video_class_train/model5_16f_2/best.pt')
    )


    def get_closest_id(x: float, data: list[tuple, ...]) -> int:
        dist = [(abs(data[i][1] - x), i) for i in range(len(data))]
        dist = sorted(dist)
        # print("Dist", dist)
        return dist[0][1]


    stop_flag = False
    count = 0
    last_track_seq = {'tr1': [], 'tr2': []}
    for i in range(0, finish):
        _, img1 = min_vc.read()

        closest_id = get_closest_id(min_range[0][1], max_range[:10])
        min_range.pop(0)
        ids = list(range(closest_id)) if closest_id else [0]
        ids = sorted(ids, reverse=True)
        for id in ids:
            max_range.pop(id)
            _, img2 = max_vc.read()

        if i >= start:

            if i == finish - 1:
                stop_flag = True

            boxes_1 = true_bb_1[i]
            # print(img1.shape[:2], img2.shape[:2])
            tracker_1.process(
                frame_id=i, boxes=boxes_1, img_shape=(w1, h1), stop_flag=stop_flag, debug=False)
            boxes_2 = true_bb_2[i]
            tracker_2.process(
                frame_id=i, boxes=boxes_2, img_shape=(w2, h2), stop_flag=stop_flag, debug=False)
            print('================================================================')
            print(f"Current_frame = {i}, current count = {count}")
            print(f"Input boxes, tracker 1 = {boxes_1}, tracker 2 = {boxes_2}")
            print(f'tracker_1.track_list. Track num = {len(tracker_1.track_list)}')
            for tr in tracker_1.track_list:
                print(f"--ID={tr['id']}, frames={tr['frame_id']}, check_out={tr['check_out']}, "
                      # f"check_in={tr['check_in']}, "
                      f"boxes={tr['boxes']}")
            print('tracker_1.dead_boxes', tracker_1.dead_boxes)
            print(f'tracker_2.track_list. Track num = {len(tracker_2.track_list)}')
            for tr in tracker_2.track_list:
                print(f"--ID={tr['id']}, frames={tr['frame_id']}, check_out={tr['check_out']}, "
                      # f"check_in={tr['check_in']}, "
                      f"boxes={tr['boxes']}")
            print('tracker_2.dead_boxes', tracker_2.dead_boxes)
            existing_tracks = [len(tracker_1.track_list), len(tracker_2.track_list)]

            class_counter, last_track_seq = PolyTracker.combine_count(
                count=count,
                last_track_seq=last_track_seq,
                tracker_1_count_frames=copy.deepcopy(tracker_1.count_frames),
                tracker_2_count_frames=copy.deepcopy(tracker_2.count_frames),
                frame_id=i,
                class_counter=class_counter,
                class_list=classes,
                class_model=vc.model,
                existing_tracks=existing_tracks,
                stop_flag=stop_flag,
                debug=True,
                frame_size=vc.frame_size
            )
            count = len(class_counter)
            cl_count = {cl: 0 for cl in classes}
            # print('class_counter', class_counter)
            if class_counter:
                cl_count.update(dict(Counter(class_counter)))
                # print(dict(Counter(class_counter)), cl_count)
            print('================================================================')
            # Draw all figures on image
            img1 = PolyTracker.prepare_image(
                image=img1,
                colors=colors,
                tracker_current_boxes=tracker_1.current_boxes,
                # polygon_in=tracker_1.polygon_in,
                polygon_out=tracker_1.polygon_out,
                poly_width=5,
                reshape=out_size
            )
            # cv2.imshow('image', img1)
            # cv2.waitKey(0)
            img2 = PolyTracker.prepare_image(
                image=img2,
                colors=colors,
                tracker_current_boxes=tracker_2.current_boxes,
                # polygon_in=tracker_2.polygon_in,
                polygon_out=tracker_2.polygon_out,
                poly_width=2,
                reshape=out_size
            )
            # cv2.imshow('image', img2)
            # cv2.waitKey(0)

            img = np.concatenate((img1, img2), axis=0)
            txt = ''
            for cl in classes:
                txt = f"{txt}{cl} - {cl_count.get(cl)}\n"
            headline = f"Обнаружено ковров: {count}\n" \
                       f"{txt[:-1]}"
            # f"Трекер 1: {tracker_1.count}\n" \
            # f"Трекер 2: {tracker_2.count}\n" \
            img = add_headline_to_cv_image(
                image=img,
                headline=headline
            )
            # cv_img = cv2.cvtColor(img)
            cv2.imshow('image', img)
            cv2.waitKey(1)

            if (i + 1) % 100 == 0:
                logger.info(f"Frames {i + 1} / {finish} was processed. Current count: {count}")
            out.write(img)

            # if count:
            #     break

            # break
    logger.info(f"\nFinal count={count}")
    out.release()
# 115x200 1264.mp4 camera_1
# 24 [[0.2604166666666667, 0.37777777777777777, 0.39895833333333336, 0.4648148148148148], [0.26875, 0.38333333333333336, 0.409375, 0.4703703703703704], [0.29270833333333335, 0.387037037037037, 0.43020833333333336, 0.4981481481481482], [0.29270833333333335, 0.387037037037037, 0.43020833333333336, 0.4981481481481482], [0.3203125, 0.375, 0.440625, 0.5], [0.36875, 0.362962962962963, 0.4609375, 0.48518518518518516], [0.38177083333333334, 0.3425925925925926, 0.471875, 0.5212962962962963], [0.4375, 0.3, 0.5015625, 0.46574074074074073], [0.43802083333333336, 0.3, 0.5010416666666667, 0.46574074074074073], [0.4505208333333333, 0.2722222222222222, 0.5072916666666667, 0.42777777777777776], [0.4515625, 0.2462962962962963, 0.5276041666666667, 0.36944444444444446], [0.45208333333333334, 0.2222222222222222, 0.5463541666666667, 0.33425925925925926], [0.4791666666666667, 0.16944444444444445, 0.5854166666666667, 0.26481481481481484], [0.4791666666666667, 0.16944444444444445, 0.5854166666666667, 0.26481481481481484], [0.49635416666666665, 0.15, 0.6010416666666667, 0.25462962962962965], [0.5208333333333334, 0.13055555555555556, 0.6208333333333333, 0.24259259259259258], [0.5109375, 0.12037037037037036, 0.6348958333333333, 0.21388888888888888], [0.5270833333333333, 0.10185185185185185, 0.6609375, 0.20092592592592592], [0.5270833333333333, 0.10185185185185185, 0.6609375, 0.20092592592592592], [0.5390625, 0.09444444444444444, 0.6666666666666666, 0.18981481481481483], [0.553125, 0.08518518518518518, 0.6776041666666667, 0.18611111111111112], [0.5645833333333333, 0.0787037037037037, 0.6864583333333333, 0.19166666666666668], [0.5927083333333333, 0.06481481481481481, 0.7010416666666667, 0.18888888888888888], [0.5927083333333333, 0.06481481481481481, 0.7010416666666667, 0.18888888888888888]]
# 24 [[], [], [], [], [0.215625, 0.4888888888888889, 0.284375, 0.6583333333333333], [0.1953125, 0.4888888888888889, 0.2953125, 0.6722222222222223], [0.184375, 0.48333333333333334, 0.3078125, 0.6833333333333333], [0.175, 0.475, 0.31875, 0.675], [0.1875, 0.44722222222222224, 0.3296875, 0.7055555555555556], [0.2015625, 0.41388888888888886, 0.3453125, 0.7], [0.2296875, 0.3972222222222222, 0.365625, 0.7], [0.2546875, 0.3861111111111111, 0.378125, 0.6944444444444444], [0.284375, 0.3972222222222222, 0.4, 0.7333333333333333], [0.2984375, 0.4, 0.4125, 0.7527777777777778], [0.3296875, 0.4083333333333333, 0.4578125, 0.7722222222222223], [0.375, 0.4111111111111111, 0.503125, 0.8], [0.421875, 0.4083333333333333, 0.55, 0.8194444444444444], [0.4625, 0.4083333333333333, 0.621875, 0.8527777777777777], [0.490625, 0.41388888888888886, 0.653125, 0.85], [0.503125, 0.43333333333333335, 0.7015625, 0.8861111111111111], [0.515625, 0.4666666666666667, 0.7453125, 0.9305555555555556], [], [], []]
