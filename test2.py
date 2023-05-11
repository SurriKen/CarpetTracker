import numpy as np

from parameters import IMAGE_IRRELEVANT_SPACE_PERCENT, MIN_OBJ_SEQUENCE
from test import tracker_1_coord, tracker_2_coord


class Tracker:
    def __init__(self):
        self.tracker_dict = {}
        self.working_idxs = []
        self.last_relevant_frames = []
        self.empty_seq = []
        self.carpet_seq = []
        self.dead_boxes = {}
        self.count = 0
        pass

    @staticmethod
    def remove_perimeter_boxes(box: list, origin_shape: tuple):
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
    def get_distance(box1: list, box2: list):
        # print(box1, box2)
        c1 = (int((box1[0] + box1[2]) / 2), int((box1[1] + box1[3]) / 2))
        c2 = (int((box2[0] + box2[2]) / 2), int((box2[1] + box2[3]) / 2))
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    # @staticmethod
    def fill_new_id_form(self, new_id):
        self.tracker_dict[new_id] = {
            'frame_id': [],  # 10, 20, 30
            'coords': [],
            'speed': [],
            # 'coords_2': [],
            'relevant': True,  # False
            # 'status': 'run',  # 'end'
            # 'images': [],  # arrays from 2 cameras
            # 'carpet_size': '',
        }
        # return id_dict

    # @staticmethod
    def update_id(
            self, id: int, frame_id: int, coords_1: list,
            # coords_2: list, images,
            relevant: bool = True,
            # status: str = 'run', carpet_size: str = ''
            ):
        if id not in list(self.tracker_dict.keys()):
            self.fill_new_id_form(
                new_id=id,
                # id_dict=id_dict
            )
        self.tracker_dict[id]['frame_id'].append(frame_id)
        if coords_1:
            self.tracker_dict[id]['coords'].append(coords_1)
        # if coords_2:
        #     id_dict[id]['coords_2'].append(coords_2)
        # id_dict[id]['images'].append(images)
        if not relevant:
            self.tracker_dict[id]['relevant'] = relevant
            # id_dict[id]['images'] = []
        # if status == 'end':
        #     id_dict[id]['status'] = status
        #     id_dict[id]['images'] = []
        # if carpet_size:
        #     id_dict[id]['carpet_size'] = carpet_size
        # return id_dict

    def fill_dead_box_form(self, key, id, box):
        self.dead_boxes[key]['id'].append(id)
        self.dead_boxes[key]['boxes'].append(box)

    def move_boxes_from_track_to_dead(self, frame_idxs: list, boxes: list):
        max_id = max(list(self.dead_boxes.keys()))
        self.dead_boxes[max_id+1] = {
            'id': frame_idxs,
            'box': boxes
        }

    def update_dead_boxes(self, new_id, new_box, distance_limit):
        drop_keys = []
        find = False
        for k in self.dead_boxes.keys():
            if new_id - self.dead_boxes.get(k).get('id')[-1] > MIN_OBJ_SEQUENCE:
                drop_keys.append(k)
            elif self.get_distance(self.dead_boxes.get(k).get('id')[-1], new_box) < distance_limit:
                self.fill_dead_box_form(key=k, id=new_id, box=new_box)
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
        boxes_for_track = {}
        for i, b in enumerate(boxes):
            boxes_for_track[i] = b

        cur_boxes = {}
        for i in working_idxs:
            cur_boxes[i] = track_dict[i][-1]

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

    def process_camera_1(self, frame_id, predict_camera_1, remove_perimeter_boxes=True):
        bb1 = predict_camera_1
        origin_shape_1 = (1080, 1920)
        speed_limit = 0.02 * (origin_shape_1[0] ** 2 + origin_shape_1[1] ** 2) ** 0.5

        # Remove perimeter irrelevant boxes if condition is True
        if bb1:
            bb1 = [Tracker.remove_perimeter_boxes(bb, origin_shape_1) for bb in bb1] \
                if remove_perimeter_boxes else bb1

        # Check in dead boxes list and update it
        if bb1 and self.dead_boxes:
            bb1_upd = []
            for box in bb1:
                box = self.update_dead_boxes(new_id=frame_id, new_box=box, distance_limit=speed_limit)
                if box:
                    bb1_upd.append(box)
            bb1 = bb1_upd

        # Track boxes
        if bb1:
            tracked_boxes, untracked_boxes = self.track_box(
                boxes=bb1,
                track_dict=self.tracker_dict,
                working_idxs=self.working_idxs
            )

        # write in track list if bb1 not empty






    def process(self, frame_id, predict_camera_1, predict_camera_2, remove_perimeter_boxes=None):
        if remove_perimeter_boxes is None:
            remove_perimeter_boxes = [True, False]
        bb1, bb2 = predict_camera_1, predict_camera_2
        origin_shape_1, origin_shape_2 = (1080, 1920), (360, 640)

        # Remove perimeter irrelevant boxes if condition is True
        bb1 = [Tracker.remove_perimeter_boxes(bb, origin_shape_1) for bb in bb1] if bb1 and remove_perimeter_boxes[0] else bb1
        bb2 = [Tracker.remove_perimeter_boxes(bb, origin_shape_2) for bb in bb2] if bb2 and remove_perimeter_boxes[1] else bb2
        empty, carpet = self.sequence_update(frame_id, bb1, bb2)

        # Identify box id
        # if bb1:
        #     tracked_boxes_1, unused_id_keys_1, unused_boxes_1 = Tracker.track_box(
        #         boxes=bb1, track_dict=self.tracker_dict, working_idxs=self.working_idxs, mode='coords_1'
        #     )
        # if bb2:
        #     tracked_boxes_2, unused_id_keys_2, unused_boxes_2 = Tracker.track_box(
        #         boxes=bb2, track_dict=self.tracker_dict, working_idxs=self.working_idxs, mode='coords_2'
        #     )

        # Remove dead boxes

        # Identify box id and fill track dictionary
        if bb1 or bb2 and not self.working_idxs:
            max_id = max(list(self.tracker_dict.keys())) if self.tracker_dict else 0
            idx = list(range(max_id + 1, max([len(bb1), len(bb2)]) + max_id + 1))
            self.working_idxs.extend(idx)
            for i in idx:
                self.tracker_dict = Tracker.update_id(
                    id_dict=self.tracker_dict,
                    id=i,
                    frame_id=frame_id,
                    coords_1=bb1[i] if i + 1 <= len(bb1) else [],
                    coords_2=bb2[i] if i + 1 <= len(bb2) else [],
                    images=None,
                )
        elif bb1 or bb2:
            max_id = max(list(self.tracker_dict.keys())) if self.tracker_dict else 0
            idx = list(range(max_id+1, max([len(bb1), len(bb2)])+max_id+1))
            # tracked_boxes_1, unused_id_keys_1, unused_boxes_1 = Tracker.track_box(
            #     boxes=bb1, track_dict=self.tracker_dict, working_idxs=self.working_idxs, mode='coords_1'
            # )
            # tracked_boxes_2, unused_id_keys_2, unused_boxes_2 = Tracker.track_box(
            #     boxes=bb2, track_dict=self.tracker_dict, working_idxs=self.working_idxs, mode='coords_1'
            # )
            self.working_idxs.extend(idx)
            for i in idx:
                self.tracker_dict = Tracker.update_id(
                    id_dict=self.tracker_dict,
                    id=i,
                    frame_id=frame_id,
                    coords_1=bb1[i] if i + 1 <= len(bb1) else [],
                    coords_2=bb2[i] if i + 1 <= len(bb2) else [],
                    images=None,
                )

        # Fill track dictionary

        # if not self.working_ids:
        #

        # if bb1 or bb2:
        #
        #     else:
        #         id_bb = []
        #         # check if is boxes related to working id
        #         # if
        #         pass
        pass


if __name__ == "__main__":
    tr = Tracker()
    for i in range(len(tracker_1_coord)):
        if len(tracker_1_coord[i]) > 1:
            print(i, tracker_1_coord[i])
    #     tr.process(
    #         frame_id=i,
    #         predict_camera_1=tracker_1_coord[i],
    #         predict_camera_2=tracker_2_coord[i],
    #     )
    #     if i > 1500:
    #         es = tr.empty_seq[-MIN_OBJ_SEQUENCE:] if len(tr.empty_seq) >= MIN_OBJ_SEQUENCE else tr.empty_seq
    #         cs = tr.carpet_seq[-MIN_OBJ_SEQUENCE:] if len(tr.carpet_seq) >= MIN_OBJ_SEQUENCE else tr.carpet_seq
    #         # print(i, 'empty_seq=', es, 'carpet_seq=', cs, 'predict=', tracker_1_coord[i], tracker_2_coord[i])
    #         print(i, 'predict=', tracker_1_coord[i], tracker_2_coord[i])
    #     if i == 2000:
    #         break
    # dead_track = tracker_1_coord[1549:1564]
    #
    # # print(dead_track)
    # # for tr in dead_track:
    # #     print(tr)
    # img_shape = (1080, 1920)
    # limit = 0.01 * (img_shape[0] ** 2 + img_shape[1] ** 2) ** 0.5
    # print('limit=', limit)
    # def detect_dead_track(track: list):
    #     sp = []
    #     for i in range(len(track) - 1):
    #         speed = Tracker.get_distance(track[i][0], track[i+1][0])
    #         # print(speed)
    #         sp.append(speed)
    #     if np.average(sp[:-10]) < limit:
    #         return True
    #     else:
    #         return False
    # print(detect_dead_track(dead_track))
