import copy
import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
import matplotlib.path as mpltPath
from classification.nn_classificator import VideoClassifier
from parameters import *

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
    def prepare_image(image: np.ndarray, colors: list, tracker_current_boxes: list, poly_width: int,
                      reshape: tuple, polygon_in: list, polygon_out: list) -> np.ndarray:
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
    def draw_polygons(polygons: list, image: np.ndarray, outline: tuple = (0, 200, 0), width: int = 5) -> np.ndarray:
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
        """
        Check if point is in polygon
        """
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
    def put_box_on_image(
            image: np.ndarray, labels: list, color_list: list, coordinates: list, save_path: str = None
    ) -> np.ndarray:
        image = np.transpose(image, (2, 0, 1))
        w, h = image.shape[:2]
        image = torch.from_numpy(image)
        coord = []
        for box in coordinates:
            coord.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        bbox = torch.tensor(coord, dtype=torch.int)
        if bbox.tolist():
            image_true = draw_bounding_boxes(
                image=image, boxes=bbox, width=3, labels=labels, colors=color_list, fill=True,
                font=os.path.join(ROOT_DIR, "arial.ttf"), font_size=int(h * 0.02))
            image = torchvision.transforms.ToPILImage()(image_true)
        else:
            image = torchvision.transforms.ToPILImage()(image)
        if save_path:
            image.save(f'{save_path}')
        return np.array(image)

    @staticmethod
    def add_track() -> dict:
        return dict(id=None, boxes=[], frame_id=[], check_in=[], check_out=[], shift_center=[], speed=[],
                    shift_top_left=[], shift_top_right=[], shift_bottom_left=[], shift_bottom_right=[])

    @staticmethod
    def fill_track(track: dict, id: int, frame_id: int, box: list, check_in: bool, check_out: bool) -> dict:
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
    def expand_poly(poly: list, step: int) -> list:
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

    def update_dead_boxes(self, frame_id: int, new_box: list, distance_limit: float) -> list:
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

    def fill_dead_box_form(self, key: int, frame_id: int, box: list) -> None:
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
    def predict_track_class(model: VideoClassifier, tracks: dict, classes: list, frame_size: tuple[int, int]) -> list:

        arr = model.track_to_array(
            tracks=[tracks['tr1'], tracks['tr2']], frame_size=frame_size, num_frames=model.input_size[0]
        )
        return model.predict(arr, model=model.model, classes=classes)

    @staticmethod
    def combine_count(frame_id: int, count: int, last_track_seq: dict, class_counter: list, class_list: list,
                      tracker_1_count_frames: list, tracker_2_count_frames: list, class_model: VideoClassifier,
                      existing_tracks: list[int, int], frame_size: tuple[int, int],
                      debug: bool = False, stop_flag: bool = False) -> (list, dict, list):
        if debug:
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
        end_track = {'tr1': [], 'tr2': []}
        if stop_flag and (new_track_seq['tr1'] or new_track_seq['tr2']):
            len_1 = len(new_track_seq['tr1'][0]) if new_track_seq['tr1'] else 0
            len_2 = len(new_track_seq['tr2'][0]) if new_track_seq['tr2'] else 0
            if len_1 >= MIN_OBJ_SEQUENCE or len_2 >= MIN_OBJ_SEQUENCE:
                last_track_seq['tr1'] = new_track_seq['tr1'] if new_state[0] else last_track_seq['tr1']
                last_track_seq['tr2'] = new_track_seq['tr2'] if new_state[1] else last_track_seq['tr2']
                predict_class = PolyTracker.predict_track_class(
                    model=class_model, tracks=last_track_seq, classes=class_list, frame_size=frame_size)
                class_counter.append(predict_class[0])
                end_track = copy.deepcopy(last_track_seq)
                return class_counter, last_track_seq, end_track

        if new_state == [False, False] and last_state == [False, False]:
            return class_counter, last_track_seq, end_track

        elif new_state == [False, False] and last_state != [False, False]:
            max_last_1 = max(last_track_seq['tr1'][0]) if last_track_seq['tr1'] else 0
            max_last_2 = max(last_track_seq['tr2'][0]) if last_track_seq['tr2'] else 0

            if (last_state == [True, True] and frame_id - max([max_last_1, max_last_2]) > limit) or \
                    (last_state != [False, False] and last_state != [True, True] and existing_tracks == [0, 0]):
                predict_class = PolyTracker.predict_track_class(
                    model=class_model, tracks=last_track_seq, classes=class_list, frame_size=frame_size)
                class_counter.append(predict_class[0])
                end_track = copy.deepcopy(last_track_seq)
                last_track_seq['tr1'] = []
                last_track_seq['tr2'] = []
                return class_counter, last_track_seq, end_track
            return class_counter, last_track_seq, end_track

        elif new_state != [False, False] and last_state == [False, False]:
            if new_state[0]:
                last_track_seq['tr1'] = new_track_seq['tr1']
            if new_state[1]:
                last_track_seq['tr2'] = new_track_seq['tr2']
            return class_counter, last_track_seq, end_track

        elif last_state == [True, True] or new_state == [True, True] or \
                (last_state == [True, False] and new_state == [True, False]) or \
                (last_state == [False, True] and new_state == [False, True]):
            predict_class = PolyTracker.predict_track_class(
                model=class_model, tracks=last_track_seq, classes=class_list, frame_size=frame_size)
            class_counter.append(predict_class[0])
            end_track = copy.deepcopy(last_track_seq)
            last_track_seq['tr1'] = new_track_seq['tr1'] if new_state[0] else []
            last_track_seq['tr2'] = new_track_seq['tr2'] if new_state[1] else []
            return class_counter, last_track_seq, end_track

        else:
            last_track_seq['tr1'] = new_track_seq['tr1'] if new_state[0] else last_track_seq['tr1']
            last_track_seq['tr2'] = new_track_seq['tr2'] if new_state[1] else last_track_seq['tr2']
            return class_counter, last_track_seq, end_track

    @staticmethod
    def isin(pattern: list, sequence: list) -> bool:
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i + len(pattern)] == pattern:
                return True
        return False

    def check_closest_track(self, check_id):
        check_track = self.track_list[check_id]
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
            id = closest_id[0][1]

        if id is not None:
            min_frame = min([min(check_track['frame_id']), min(self.track_list[id]['frame_id'])])
            max_frame = max([max(check_track['frame_id']), max(self.track_list[id]['frame_id'])])
            new_track = self.add_track()
            for fr in range(min_frame, max_frame):
                if fr in self.track_list[id]['frame_id'] and fr in check_track['frame_id']:
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
                            check_in=self.track_list[id]['check_in'][self.track_list[id]['frame_id'].index(fr)],
                            check_out=self.track_list[id]['check_out'][self.track_list[id]['frame_id'].index(fr)]
                        )
                    else:
                        new_track = self.fill_track(
                            track=new_track,
                            id=self.track_list[id]['id'],
                            frame_id=fr,
                            box=check_track['boxes'][check_track['frame_id'].index(fr)],
                            check_out=check_track['check_out'][check_track['frame_id'].index(fr)],
                            check_in=check_track['check_in'][check_track['frame_id'].index(fr)],
                        )
                elif fr in self.track_list[id]['frame_id']:
                    new_track = self.fill_track(
                        track=new_track,
                        id=self.track_list[id]['id'],
                        frame_id=fr,
                        box=self.track_list[id]['boxes'][self.track_list[id]['frame_id'].index(fr)],
                        check_in=self.track_list[id]['check_in'][self.track_list[id]['frame_id'].index(fr)],
                        check_out=self.track_list[id]['check_out'][self.track_list[id]['frame_id'].index(fr)]
                    )
                elif fr in check_track['frame_id']:
                    new_track = self.fill_track(
                        track=new_track,
                        id=self.track_list[id]['id'],
                        frame_id=fr,
                        box=check_track['boxes'][check_track['frame_id'].index(fr)],
                        check_in=check_track['check_in'][check_track['frame_id'].index(fr)],
                        check_out=check_track['check_out'][check_track['frame_id'].index(fr)]
                    )

            self.track_list[id] = new_track
            self.track_list.pop(check_id)

    def update_track_list(
            self, distance_limit: float, img_shape: tuple, stop_flag: bool = False, debug: bool = False) -> None:
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

        max_len = 0
        for i in deleted:
            if max_len < len(self.track_list[i]['frame_id']):
                self.count_frames = [
                    self.track_list[i]['frame_id'],
                    [[b / img_shape[i % 2] for i, b in enumerate(bb)] for bb in self.track_list[i]['boxes']]
                ]
        self.track_list = [self.track_list[i] for i in rel]

    def process(self, frame_id: int, boxes: list, img_shape: tuple, speed_limit_percent: float = SPEED_LIMIT_PERCENT,
                stop_flag: bool = False, debug: bool = False) -> None:
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

            # if not self.point_in_polygon(center, limit_in) and self.point_in_polygon(center, limit_out):
            if self.point_in_polygon(center, limit_out):  # and not self.point_in_polygon(center, limit_in):
                check_in = self.point_in_polygon(center, limit_in)
                check_out = self.point_in_polygon(center, self.polygon_out)

                # Check in dead boxes list and update it
                if self.dead_boxes:
                    box = self.update_dead_boxes(frame_id=frame_id, new_box=box, distance_limit=dist_limit)

                if box:
                    self.current_boxes.append([box, check_in, check_out])

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

        self.update_track_list(distance_limit=dist_limit, img_shape=img_shape, debug=debug, stop_flag=stop_flag)
        if debug:
            print('self.dead_boxes', self.dead_boxes)
