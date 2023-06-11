import copy
import os

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import matplotlib.path as mpltPath

from dataset_processing import DatasetProcessing
from parameters import IMAGE_IRRELEVANT_SPACE_PERCENT, MIN_OBJ_SEQUENCE, MIN_EMPTY_SEQUENCE, SPEED_LIMIT_PERCENT, \
    POLY_CAM1_IN, POLY_CAM1_OUT, POLY_CAM2_IN, POLY_CAM2_OUT, GLOBAL_STEP, ROOT_DIR


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
    def put_box_on_image(save_path, image, labels, color_list, coordinates):
        # image = results[0].orig_img[:, :, ::-1].copy()
        # print(image.shape)
        # if draw_poly:
        #     image = Image.fromarray(image)
        #     poly = [POLY_CAM1_IN, POLY_CAM1_OUT] if camera == 1 else [POLY_CAM2_IN, POLY_CAM2_OUT]
        #     image = DatasetProcessing.draw_polygons(polygons=poly[0], image=image, outline='green')
        #     image = DatasetProcessing.draw_polygons(polygons=poly[1], image=image, outline='red')
        #     # image.show()
        #     image = np.array(image)
        #     # image = np.expand_dims(image, axis=0)
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
    def get_distance(box1: list, box2: list) -> tuple[float, float, float, float, float]:
        """
        :param box1: list of coordinates with first 4 index positions [x1, y1, x2, y2]
        :param box2: list of coordinates with first 4 index positions [x1, y1, x2, y2]
        :return: float distance between the two box centers
        """
        # print(box1, box2)
        c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        shift_center = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        shift_top_left = ((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2) ** 0.5
        shift_top_right = ((box1[0] - box2[0]) ** 2 + (box1[3] - box2[3]) ** 2) ** 0.5
        shift_bottom_left = ((box1[2] - box2[2]) ** 2 + (box1[1] - box2[1]) ** 2) ** 0.5
        shift_bottom_right = ((box1[2] - box2[2]) ** 2 + (box1[3] - box2[3]) ** 2) ** 0.5
        return shift_center, shift_top_left, shift_top_right, shift_bottom_left, shift_bottom_right

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
            'shift_center': [],
            'shift_top_left': [],
            'shift_top_right': [],
            'shift_bottom_left': [],
            'shift_bottom_right': [],
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

    @staticmethod
    def get_distribution(pattern, tracker_1_dict, tracker_2_dict) -> dict:
        keys1 = list(tracker_1_dict.keys())
        keys2 = list(tracker_2_dict.keys())
        dist = {}
        # pat_for_analysis = []
        for i, pat in enumerate(pattern):
            dist[i] = {'cam1': [], 'cam2': [], 'tr_num': 0, 'sequence': pat}
            if keys1:
                for k in tracker_1_dict.keys():
                    if min(tracker_1_dict.get(k).get('frame_id')) in pat and max(
                            tracker_1_dict.get(k).get('frame_id')) in pat:
                        dist[i]['cam1'].append(k)
                        dist[i]['tr_num'] = len(dist[i]['cam1']) if dist[i]['tr_num'] < len(dist[i]['cam1']) else \
                            dist[i][
                                'tr_num']
                        keys1.pop(keys1.index(k))
            if keys2:
                for k in tracker_2_dict.keys():
                    if min(tracker_2_dict.get(k).get('frame_id')) in pat and max(
                            tracker_2_dict.get(k).get('frame_id')) in pat:
                        dist[i]['cam2'].append(k)
                        dist[i]['tr_num'] = len(dist[i]['cam2']) if dist[i]['tr_num'] < len(dist[i]['cam2']) else \
                            dist[i][
                                'tr_num']
                        keys2.pop(keys2.index(k))
            # if dist[i]['tr_num'] > 1:
            #     pat_for_analysis.append(i)
        return dist

    @staticmethod
    def get_center(coord: list[int, int, int, int]) -> tuple[int, int]:
        return int((coord[0] + coord[2]) / 2), int((coord[1] + coord[3]) / 2)

    @staticmethod
    def tracker_analysis(pattern, main_track, main_cam):
        vecs = {}
        for id in pattern.get(main_cam):
            start = [int(i) for i in main_track.get(id).get('coords')[0][:4]]
            fin = [int(i) for i in main_track.get(id).get('coords')[-1][:4]]
            vector_x = Tracker.get_center(fin)[0] - Tracker.get_center(start)[0]
            vecs[id] = {'start': start, 'fin': fin, 'vector_x': vector_x,
                        'frame_start': main_track.get(id).get('frame_id')[0],
                        'frame_fin': main_track.get(id).get('frame_id')[-1]}

        for id in vecs.keys():
            vecs[id]['cross_vecs'] = {}
            vecs[id]['follow_id'] = []
            for id2 in vecs.keys():
                if id2 != id:
                    vecs[id]['cross_vecs'][id2] = \
                        Tracker.get_center(vecs[id2]['start'])[0] - Tracker.get_center(vecs[id]['fin'])[0]
                    if vecs[id]['cross_vecs'][id2] > 0 and vecs[id2]['frame_start'] > vecs[id]['frame_fin']:
                        vecs[id]['follow_id'].append(id2)

        uniq = list(vecs.keys())
        follow = []
        for id in vecs.keys():
            # print(id, vecs[id])
            if vecs[id]['follow_id']:
                for id2 in vecs[id]['follow_id']:
                    if id2 in uniq:
                        uniq.pop(uniq.index(id2))
                    if id2 not in follow:
                        follow.append(id2)

        new_patterns = []
        for id in uniq:
            new_vec = {
                'track_id': id,
                'start': vecs[id]['start'],
                'frame_idx': copy.deepcopy(main_track.get(id).get('frame_id')),
            }
            if vecs[id]['follow_id']:
                for v in vecs[id]['follow_id']:
                    new_vec['frame_idx'].extend(main_track.get(v).get('frame_id'))
                new_vec['frame_idx'] = sorted(list(set(new_vec['frame_idx'])))
            new_patterns.append(new_vec)
        return new_patterns

    @staticmethod
    def combine_res(main_res, sec_res):
        res = {main_res[id]['track_id']: {'seq_id': [], 'frame_idx': main_res[id]['frame_idx']} for id in
               range(len(main_res))}

        frame_dist = []
        for sid in range(len(sec_res)):
            for mid in range(len(main_res)):
                frame_dist.append(
                    (abs(main_res[mid]['frame_idx'][0] - sec_res[sid]['frame_idx'][0]), mid, sid))
        frame_dist = sorted(frame_dist)
        m_key = [i for i in range(len(main_res))]
        s_key = [i for i in range(len(sec_res))]

        for fr in frame_dist:
            if fr[1] in m_key and fr[2] in s_key:
                # res[fr[1]].append(fr[2])
                res[main_res[fr[1]]['track_id']]['seq_id'].append(sec_res[fr[2]]['track_id'])
                res[main_res[fr[1]]['track_id']]['frame_idx'].extend(sec_res[fr[2]]['frame_idx'])
                res[main_res[fr[1]]['track_id']]['frame_idx'] = sorted(
                    list(set(res[main_res[fr[1]]['track_id']]['frame_idx'])))
                m_key.pop(m_key.index(fr[1]))
                s_key.pop(s_key.index(fr[2]))
            else:
                continue

            if not s_key:
                break
        return res

    @staticmethod
    def pattern_analisys(pattern: dict, tracker_1_dict: dict, tracker_2_dict: dict):
        if pattern.get('tr_num') == 1:
            return [pattern]
        else:
            if len(pattern.get('cam1')) > 1:
                res_1 = Tracker.tracker_analysis(
                    pattern=pattern,
                    main_track=tracker_1_dict,
                    main_cam='cam1'
                )
            elif not pattern.get('cam1'):
                res_1 = {}
            else:
                res_1 = [{
                    'track_id': pattern.get('cam1')[0],
                    'start': [int(i) for i in tracker_1_dict.get(pattern.get('cam1')[0]).get('coords')[0][:4]],
                    'frame_idx': copy.deepcopy(tracker_1_dict.get(pattern.get('cam1')[0]).get('frame_id')),
                }]
            # print('res_1: {0}'.format(res_1))

            if len(pattern.get('cam2')) > 1:
                res_2 = Tracker.tracker_analysis(
                    pattern=pattern,
                    main_track=tracker_2_dict,
                    main_cam='cam2'
                )
            elif not pattern.get('cam2'):
                res_2 = {}
            else:
                res_2 = [{
                    'track_id': pattern.get('cam2')[0],
                    'start': [int(i) for i in tracker_2_dict.get(pattern.get('cam2')[0]).get('coords')[0][:4]],
                    'frame_idx': copy.deepcopy(tracker_2_dict.get(pattern.get('cam2')[0]).get('frame_id')),
                }]
            # print('res_2: {0}'.format(res_2))

            if not res_2:
                return [
                    {'cam1': [res_1[i]['track_id']], 'cam2': [], 'tr_num': 1,
                     'sequence': res_1[i]['frame_idx']} for i in range(len(res_1))
                ]
            if not res_1:
                return [
                    {'cam1': [], 'cam2': [res_2[i]['track_id']], 'tr_num': 1,
                     'sequence': res_2[i]['frame_idx']} for i in range(len(res_2))
                ]

            main_res = res_1 if len(res_1) > len(res_2) else res_2
            sec_res = res_2 if main_res == res_1 else res_1

            res = Tracker.combine_res(main_res, sec_res)
            # print(res)
            return [
                {'cam1': [id], 'cam2': res[id]['seq_id'], 'tr_num': 1,
                 'sequence': res[id]['frame_idx']} for id in res.keys()
            ]

    @staticmethod
    def clean_tracks(frame, pattern, tracker_1_dict, tracker_2_dict):
        old_pattern_count = 0
        # print('pattern', pattern)
        relevant, not_rel = [], []
        for i, pat in enumerate(pattern):
            if frame - pat[-1] <= 2 * MIN_EMPTY_SEQUENCE:
                relevant.append(i)
            else:
                not_rel.append(i)
                old_pattern_count += 1
        # print('relevant', relevant)
        rel_pattern = []
        if relevant:
            rel_pattern = np.array(pattern, dtype=object)[relevant].tolist()

        old_pat = []
        if not_rel:
            old_pat = np.array(pattern, dtype=object)[not_rel].tolist()

        remove_keys = []
        for key in tracker_1_dict.keys():
            if frame - tracker_1_dict[key]['frame_id'][-1] > 2 * MIN_EMPTY_SEQUENCE:
                remove_keys.append(key)
        for key in remove_keys:
            tracker_1_dict.pop(key)

        remove_keys = []
        for key in tracker_2_dict.keys():
            if frame - tracker_2_dict[key]['frame_id'][-1] > 2 * MIN_EMPTY_SEQUENCE:
                remove_keys.append(key)
        for key in remove_keys:
            tracker_2_dict.pop(key)

        return old_pattern_count, rel_pattern, old_pat

    @staticmethod
    def update_pattern(pattern, tracker_1_dict, tracker_2_dict) -> list:
        # x = time.time()
        dist = Tracker.get_distribution(
            pattern=pattern,
            tracker_1_dict=tracker_1_dict,
            tracker_2_dict=tracker_2_dict
        )
        # for k, v in dist.items():
        #     print(f"pat {k}: {v}")
        # print('================================================================')
        # print("time Tracker.get_distribution:", len(pattern), time_converter(time.time() - x))
        # x = time.time()
        new_pat = []
        for i in dist.keys():
            pat = Tracker.pattern_analisys(
                pattern=dist.get(i),
                tracker_1_dict=tracker_1_dict,
                tracker_2_dict=tracker_2_dict
            )
            # print(pat)
            pat = [p['sequence'] for p in pat]
            new_pat.extend(pat)
        # print("time Tracker.pattern_analisys:", len(dist), time_converter(time.time() - x))
        return new_pat

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
            shift_center, shift_top_left, shift_top_right, shift_bottom_left, shift_bottom_right = \
                self.get_distance(self.tracker_dict[id]['coords'][-2], self.tracker_dict[id]['coords'][-1])
            self.tracker_dict[id]['shift_center'].append(shift_center)
            self.tracker_dict[id]['shift_top_left'].append(shift_top_left)
            self.tracker_dict[id]['shift_top_right'].append(shift_top_right)
            self.tracker_dict[id]['shift_bottom_left'].append(shift_bottom_left)
            self.tracker_dict[id]['shift_bottom_right'].append(shift_bottom_right)
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
            elif len(self.tracker_dict[id]['shift_center']) >= MIN_OBJ_SEQUENCE and \
                    (
                            max(self.tracker_dict[id]['shift_center'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(self.tracker_dict[id]['shift_top_left'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(self.tracker_dict[id]['shift_bottom_right'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(self.tracker_dict[id]['shift_top_right'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(self.tracker_dict[id]['shift_bottom_left'][-MIN_OBJ_SEQUENCE:]) < distance_limit
                    ):
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
            # self.tracker_dict[k]['frame_id'] = sorted(list(set(self.tracker_dict[k]['frame_id'])))
            if len(self.tracker_dict[k]['coords']) >= MIN_OBJ_SEQUENCE:
                # print(frame_id, k, len(self.tracker_dict[k]['coords']), self.tracker_dict[k]['countable'],
                #       self.tracker_dict[k]['coords'])
                self.tracker_dict[k]['countable'] = True
            elif frame_id - self.tracker_dict[k]['frame_id'][-1] > MIN_EMPTY_SEQUENCE and \
                    MIN_OBJ_SEQUENCE > len(self.tracker_dict[k]['coords']) and k not in remove_keys:
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
            shift_center, shift_top_left, shift_top_right, shift_bottom_left, shift_bottom_right = \
                self.get_distance(box1=self.dead_boxes.get(k).get('coords')[-1], box2=new_box)
            if frame_id - self.dead_boxes.get(k).get('frame_id')[-1] > MIN_OBJ_SEQUENCE:
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

    def process(self, frame_id: int, predict: dict, remove_perimeter_boxes: bool = True,
                speed_limit_percent: float = SPEED_LIMIT_PERCENT):
        bb1 = predict.get('boxes')
        origin_shape_1 = predict.get('orig_shape')
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


class PolyTracker:
    def __init__(self, polygon_in: list, polygon_out: list, name: str = ''):
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
    def prepare_image(image, colors, tracker_current_boxes, poly_width, reshape, polygon_in, polygon_out):
        # Draw all figures on image
        img = PolyTracker.draw_polygons(polygons=polygon_in, image=image, outline=(0, 255, 0), width=poly_width)
        img = PolyTracker.draw_polygons(polygons=polygon_out, image=img, outline=(0, 0, 255), width=poly_width)

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

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
        return dict(id=None, boxes=[], frame_id=[], check_in=[], check_out=[], shift_center=[],
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
        drop_keys = []
        find = False
        for k in self.dead_boxes.keys():
            shift_center, shift_top_left, shift_top_right, shift_bottom_left, shift_bottom_right = \
                self.get_full_distance(box1=self.dead_boxes.get(k).get('coords')[-1], box2=new_box)
            if frame_id - self.dead_boxes.get(k).get('frame_id')[-1] > MIN_OBJ_SEQUENCE:
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
        else:
            return new_box

    def fill_dead_box_form(self, key, frame_id, box):
        """
        Fill the dead box form with data

        :param key: int inner dict id for dead track, id should already be in self.dead_boxes
        :param frame_id: int
        :param box: list of coordinates, ex. [x1, y1, x2, y2]
        """
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
    def combine_count(frame_id: int, count: int, last_track_seq: dict,
                      tracker_1_count_frames: list, tracker_2_count_frames: list) -> (int, dict):
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

        # if new_state != [False, False]:
        #     print(f"-- frame_id={frame_id}, count {count}, last_state={last_state}, new_state={new_state}, "
        #           f"last_track_seq={last_track_seq}, new_track_seq={new_track_seq}")

        if new_state == [False, False]:
            pass

        elif (last_state == [False, False] and new_state == [True, False]) or \
                (last_state == [True, False] and new_state == [True, False]):
            # print(1)
            last_track_seq['tr1'] = new_track_seq['tr1']
            count += 1

        elif (last_state == [False, False] and new_state == [False, True]) or \
                (last_state == [False, True] and new_state == [False, True]):
            # print(2)
            last_track_seq['tr2'] = new_track_seq['tr2']
            count += 1

        elif last_state == [True, False] and new_state == [False, True]:
            if min(new_track_seq['tr2']) - max(last_track_seq['tr1']) > MIN_EMPTY_SEQUENCE:
                # print(3)
                last_track_seq['tr1'] = []
                last_track_seq['tr2'] = new_track_seq['tr2']
                count += 1
            else:
                # print(4)
                last_track_seq['tr2'] = new_track_seq['tr2']

        elif last_state == [False, True] and new_state == [True, False]:
            if min(new_track_seq['tr1']) - max(last_track_seq['tr2']) > MIN_EMPTY_SEQUENCE:
                # print(5)
                last_track_seq['tr2'] = []
                last_track_seq['tr1'] = new_track_seq['tr1']
                count += 1
            else:
                # print(6)
                last_track_seq['tr1'] = new_track_seq['tr1']

        elif last_state == [True, True] and new_state == [True, False]:
            if min(new_track_seq['tr1']) - max(
                    [max(last_track_seq['tr1']), max(last_track_seq['tr2'])]) > MIN_EMPTY_SEQUENCE:
                # print(7)
                last_track_seq['tr2'] = []
                last_track_seq['tr1'] = new_track_seq['tr1']
                count += 1
            else:
                # print(8)
                last_track_seq['tr1'] = new_track_seq['tr1']

        elif last_state == [True, True] and new_state == [False, True]:
            if min(new_track_seq['tr2']) - max(
                    [max(last_track_seq['tr1']), max(last_track_seq['tr2'])]) > MIN_EMPTY_SEQUENCE:
                # print(9)
                last_track_seq['tr1'] = []
                last_track_seq['tr2'] = new_track_seq['tr2']
                count += 1
            else:
                # print(10)
                last_track_seq['tr2'] = new_track_seq['tr2']

        else:
            # print(11)
            last_track_seq['tr1'] = new_track_seq['tr1']
            last_track_seq['tr2'] = new_track_seq['tr2']
            count += 1

        return count, last_track_seq

    def update_track_list(self, distance_limit, debug=False):
        rel, deleted = [], []
        self.count_frames = []
        for i, trck in enumerate(self.track_list):
            if len(trck['shift_center']) > MIN_OBJ_SEQUENCE and \
                    (
                            max(trck['shift_center'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(trck['shift_top_left'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(trck['shift_bottom_right'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(trck['shift_top_right'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(trck['shift_bottom_left'][-MIN_OBJ_SEQUENCE:]) < distance_limit
                    ):
                self.move_boxes_from_track_to_dead(
                    frame_idxs=trck['frame_id'][-MIN_OBJ_SEQUENCE:],
                    boxes=trck['boxes'][-MIN_OBJ_SEQUENCE:]
                )
                trck['frame_id'] = trck['frame_id'][:-MIN_OBJ_SEQUENCE]
                trck['boxes'] = trck['boxes'][:-MIN_OBJ_SEQUENCE]
                trck['check_in'] = trck['check_in'][:-MIN_OBJ_SEQUENCE]
                trck['check_out'] = trck['check_out'][:-MIN_OBJ_SEQUENCE]
                trck['shift_center'] = trck['shift_center'][:-MIN_OBJ_SEQUENCE]
                trck['shift_top_left'] = trck['shift_top_left'][:-MIN_OBJ_SEQUENCE]
                trck['shift_bottom_right'] = trck['shift_bottom_right'][:-MIN_OBJ_SEQUENCE]
                trck['shift_top_right'] = trck['shift_top_right'][:-MIN_OBJ_SEQUENCE]
                trck['shift_bottom_left'] = trck['shift_bottom_left'][:-MIN_OBJ_SEQUENCE]

            if debug:
                print(trck)
            if self.frame_id - trck['frame_id'][-1] > MIN_EMPTY_SEQUENCE and trck['check_out'][-1]:
                continue
            elif 1 < len(trck['boxes']) and trck['check_out'][-2:] == [True, False]:
                deleted.append(i)
                self.count += 1
                if debug:
                    print(f"Check {self.name}", -(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1),
                              sum(trck['check_out'][:-(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]))
            # elif len(trck['boxes']) > MIN_OBJ_SEQUENCE and \
            #         sum(trck['check_out'][:-(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]) == 0:
            #     if debug:
            #         print(f"Check {self.name}", -(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1),
            #               sum(trck['check_out'][:-(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]))
            #     deleted.append(i)
            #     self.count += 1
            # elif 1 < len(trck['boxes']) <= MIN_OBJ_SEQUENCE and not trck['check_out'][-1]:
            #     if debug:
            #         print(f"Check2 {self.name}", [i['check_out'][-1] for i in self.track_list],
            #               sum(trck['check_out'][:-1]))
            #     deleted.append(i)
            #     self.count += 1
            else:
                rel.append(i)
        for i in deleted:
            self.count_frames.extend(self.track_list[i]['frame_id'])
        self.count_frames = list(set(self.count_frames))
        self.track_list = [self.track_list[i] for i in rel]

    def process(self, frame_id: int, boxes: list, img_shape: tuple, speed_limit_percent: float = SPEED_LIMIT_PERCENT,
                debug: bool = False):
        # check if boxes are relevant
        self.current_boxes = []
        self.frame_id = frame_id
        diagonal = ((img_shape[0]) ** 2 + (img_shape[1]) ** 2) ** 0.5
        speed_limit = speed_limit_percent * diagonal
        step = GLOBAL_STEP * diagonal
        limit_in = self.expand_poly(self.polygon_in, -1 * step)
        limit_out = self.expand_poly(self.polygon_out, step)
        if debug:
            print("================================")
        for box in boxes:
            center = self.get_center(box)
            # print(self.point_in_polygon(center, limit_in), self.point_in_polygon(center, limit_out))
            if not self.point_in_polygon(center, limit_in) and self.point_in_polygon(center, limit_out):
                check_in = self.point_in_polygon(center, self.polygon_in)
                check_out = self.point_in_polygon(center, self.polygon_out)

                # Check in dead boxes list and update it
                if self.dead_boxes:
                    box = self.update_dead_boxes(frame_id=frame_id, new_box=box, distance_limit=speed_limit)

                if box:
                    self.current_boxes.append([box, check_in, check_out])

        # print("track", frame_id, [[int(c) for c in b[0][:4]] for b in self.current_boxes])

        # If no track in list - write new track
        if not self.track_list:
            for box in self.current_boxes:
                if box[2]:
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
            # print("track_list 1", [i for i in self.track_list])
        # if track exist - update track
        else:
            tr_idxs = list(range(len(self.track_list)))
            box_idxs = list(range(len(self.current_boxes)))
            dist = []
            pair = []
            for i in tr_idxs:
                for b in box_idxs:
                    # distance = self.get_distance(self.track_list[i].boxes[-1], self.current_boxes[b])
                    c1 = self.track_list[i]['boxes'][-1]
                    c2 = self.current_boxes[b][0]
                    distance = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                    if (i, b) not in pair or (b, i) not in pair:
                        # if distance <= 0.05 * diagonal:
                        dist.append((distance, i, b))
                        pair.append((i, b))
            dist = sorted(dist)
            dist_limit = GLOBAL_STEP * diagonal
            # print('dist', dist, [i['boxes'][-1] for i in self.track_list], [i[0] for i in self.current_boxes])
            for d in dist:
                if tr_idxs and d[1] in tr_idxs and d[2] in box_idxs and d[0] < dist_limit:
                    # print(f"-- d: {d}")
                    self.track_list[d[1]] = self.fill_track(
                        track=self.track_list[d[1]],
                        id=self.track_list[d[1]]['id'],
                        frame_id=frame_id,
                        box=[int(c) for c in self.current_boxes[d[2]][0][:4]],
                        check_in=self.current_boxes[d[2]][1],
                        check_out=self.current_boxes[d[2]][2]
                    )
                    # self.max_id += 1
                    tr_idxs.pop(tr_idxs.index(d[1]))
                    box_idxs.pop(box_idxs.index(d[2]))
                    # print('-- tr_id', tr_idxs, 'box_id', box_idxs, self.track_list[d[1]])
            if box_idxs:
                # print("track", frame_id, [b['boxes'] for b in self.track_list])
                for b in box_idxs:
                    if self.current_boxes[b][2]:
                        # print('box_idxs', b, self.current_boxes[b])
                        # if self.point_in_polygon(self.get_center(self.current_boxes[b][0]), POLY_CAM1_OUT):
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
        #         print("track", frame_id, [b['boxes'] for b in self.track_list])
        # print("track x", self.track_list)

        # Review tracks
        if debug:
            print(
                f"track {self.name}, frame {frame_id}, count={self.count}\n",
                [i['boxes'] for i in self.track_list], '\n',
                [i['check_out'] for i in self.track_list]
            )
        self.update_track_list(distance_limit=speed_limit, debug=debug)
        # print("track x2", self.count, [i['boxes'] for i in self.track_list], [i['check_out'] for i in self.track_list])
