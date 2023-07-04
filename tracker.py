import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
import matplotlib.path as mpltPath
from parameters import MIN_OBJ_SEQUENCE, MIN_EMPTY_SEQUENCE, GLOBAL_STEP, ROOT_DIR, DEAD_LIMIT_PERCENT


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
                      tracker_1_count_frames: list, tracker_2_count_frames: list, debug=False) -> (int, dict):
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

        if new_state == [False, False]:
            pass

        elif (last_state == [False, False] and new_state == [True, False]) or \
                (last_state == [True, False] and new_state == [True, False]):
            if debug:
                print(1)
            last_track_seq['tr1'] = new_track_seq['tr1']
            count += 1

        elif (last_state == [False, False] and new_state == [False, True]) or \
                (last_state == [False, True] and new_state == [False, True]):
            if debug:
                print(2)
            last_track_seq['tr2'] = new_track_seq['tr2']
            count += 1

        elif last_state == [True, False] and new_state == [False, True]:
            if min(new_track_seq['tr2']) - max(last_track_seq['tr1']) > MIN_EMPTY_SEQUENCE:
                if debug:
                    print(3, min(new_track_seq['tr2']), max(last_track_seq['tr1']), MIN_EMPTY_SEQUENCE)
                last_track_seq['tr1'] = []
                last_track_seq['tr2'] = new_track_seq['tr2']
                count += 1
            else:
                if debug:
                    print(4, min(new_track_seq['tr2']), max(last_track_seq['tr1']), MIN_EMPTY_SEQUENCE)
                last_track_seq['tr2'] = new_track_seq['tr2']

        elif last_state == [False, True] and new_state == [True, False]:
            if min(new_track_seq['tr1']) - max(last_track_seq['tr2']) > MIN_EMPTY_SEQUENCE:
                if debug:
                    print(5, min(new_track_seq['tr1']), max(last_track_seq['tr2']), MIN_EMPTY_SEQUENCE)
                last_track_seq['tr2'] = []
                last_track_seq['tr1'] = new_track_seq['tr1']
                count += 1
            else:
                if debug:
                    print(6, min(new_track_seq['tr1']), max(last_track_seq['tr2']), MIN_EMPTY_SEQUENCE)
                last_track_seq['tr1'] = new_track_seq['tr1']

        elif last_state == [True, True] and new_state == [True, False]:
            # if min(new_track_seq['tr1']) - max(
            #         [max(last_track_seq['tr1']), max(last_track_seq['tr2'])]) > MIN_EMPTY_SEQUENCE:
            if debug:
                print(7)
            last_track_seq['tr2'] = []
            last_track_seq['tr1'] = new_track_seq['tr1']
            count += 1
            # else:
            #     print(8)
            #     last_track_seq['tr1'] = new_track_seq['tr1']

        elif last_state == [True, True] and new_state == [False, True]:
            # if min(new_track_seq['tr2']) - max(
            #         [max(last_track_seq['tr1']), max(last_track_seq['tr2'])]) > MIN_EMPTY_SEQUENCE:
            if debug:
                print(9)
            last_track_seq['tr1'] = []
            last_track_seq['tr2'] = new_track_seq['tr2']
            count += 1
            # else:
            #     print(10)
            #     last_track_seq['tr2'] = new_track_seq['tr2']

        else:
            if debug:
                print(11)
            last_track_seq['tr1'] = new_track_seq['tr1']
            last_track_seq['tr2'] = new_track_seq['tr2']
            count += 1

        return count, last_track_seq

    @staticmethod
    def isin(pattern, sequence):
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i + len(pattern)] == pattern:
                return True
        return False

    def update_track_list(self, distance_limit: float, debug: bool = False):
        rel, deleted = [], []
        self.count_frames = []
        for i, trck in enumerate(self.track_list):
            cut = MIN_OBJ_SEQUENCE - 1
            if len(trck['shift_center']) > MIN_OBJ_SEQUENCE and \
                    (
                            max(trck['shift_center'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(trck['shift_top_left'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(trck['shift_bottom_right'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(trck['shift_top_right'][-MIN_OBJ_SEQUENCE:]) < distance_limit or
                            max(trck['shift_bottom_left'][-MIN_OBJ_SEQUENCE:]) < distance_limit
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

            if debug:
                print('trck', trck)
            if self.frame_id - trck['frame_id'][-1] > MIN_EMPTY_SEQUENCE and \
                    trck['check_in'][-1]:  # or trck['check_out'][-1]):
                continue

            elif self.frame_id - trck['frame_id'][-1] > MIN_EMPTY_SEQUENCE and \
                    trck['check_out'][-1] and not self.isin([False, False, False], trck['check_out']):  # or trck['check_out'][-1]):
                continue

            elif (self.frame_id - trck['frame_id'][-1]) > MIN_EMPTY_SEQUENCE and not trck['check_out'][-1]:
                if debug:
                    print(f"Check {self.name}", -(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1),
                              sum(trck['check_out'][:-(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]))
                deleted.append(i)
                self.count += 1

            elif (self.frame_id - trck['frame_id'][-1]) > MIN_EMPTY_SEQUENCE and \
                    self.isin([False, False, False], trck['check_out']):
                if debug:
                    print(f"Check {self.name}", -(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1),
                              sum(trck['check_out'][:-(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]))
                deleted.append(i)
                self.count += 1

            else:
                rel.append(i)
        for i in deleted:
            self.count_frames.extend(self.track_list[i]['frame_id'])
        self.count_frames = list(set(self.count_frames))
        self.track_list = [self.track_list[i] for i in rel]

    def process(self, frame_id: int, boxes: list, img_shape: tuple, debug: bool = False):
        # check if boxes are relevant
        self.current_boxes = []
        self.frame_id = frame_id
        diagonal = ((img_shape[0]) ** 2 + (img_shape[1]) ** 2) ** 0.5
        # speed_limit = speed_limit_percent * diagonal
        dist_limit = DEAD_LIMIT_PERCENT * diagonal
        step = GLOBAL_STEP * diagonal
        limit_in = self.expand_poly(self.polygon_in, -1 * step * 0.5)
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
                    box = self.update_dead_boxes(frame_id=frame_id, new_box=box, distance_limit=dist_limit)

                if box:
                    self.current_boxes.append([box, check_in, check_out])

        # if debug:
        #     print("track", frame_id, [[int(c) for c in b[0][:4]] for b in self.current_boxes])

        # If no track in list - write new track
        if not self.track_list:
            for box in self.current_boxes:
                if box[2] and not box[1]:
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
                    if debug:
                        print(frame_id, track, self.track_list)
                elif not box[-1]:
                    # key = 0 if not self.dead_boxes else max(list(self.dead_boxes.keys()))
                    # self.dead_boxes[key] = dict(frame_id=[], coords=[])
                    self.move_boxes_from_track_to_dead(frame_idxs=[frame_id], boxes=[box[0]])
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
                    distance = PolyTracker.get_distance(c1, c2)
                    # speed = PolyTracker.get_distance(c1, c2) / (frame_id - self.track_list[i]['frame_id'][-1])
                    if (i, b) not in pair or (b, i) not in pair:
                        # if distance <= 0.05 * diagonal:
                        dist.append((distance, i, b))
                        pair.append((i, b))
            dist = sorted(dist)
            # if debug:
            #     print('dist =', dist,
            #           # '\nspeed_limit =', speed_limit,
            #           '\ndist_limit =', dist_limit)
            for d in dist:
                if not self.track_list[d[1]]['check_out'][-1] and self.current_boxes[d[2]][1]:
                    continue
                elif tr_idxs and d[1] in tr_idxs and d[2] in box_idxs: # and d[0] < speed_limit:
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
        if debug:
            print(
                f"track {self.name}, frame {frame_id}, count={self.count}\n"
                f"{[i['frame_id'] for i in self.track_list]}\n"
                f"{[i['boxes'] for i in self.track_list]}\n"
                f"{[i['check_out'] for i in self.track_list]}\n"
            )
        self.update_track_list(distance_limit=dist_limit, debug=debug)
        if debug:
            # print("track x2", self.count, [i['boxes'] for i in self.track_list], [i['check_out'] for i in self.track_list])
            print('self.dead_boxes', self.dead_boxes)
