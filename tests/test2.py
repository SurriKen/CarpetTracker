import copy
import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.path as mpltPath
from torchvision.utils import draw_bounding_boxes

from parameters import ROOT_DIR, MIN_EMPTY_SEQUENCE, MIN_OBJ_SEQUENCE, SPEED_LIMIT_PERCENT
from utils import load_data, time_converter, get_colors, add_headline_to_cv_image, logger

# imp1 = 'datasets/test 16_cam 1_0s-639s/frames/04424.png'
# imp2 = 'datasets/test 16_cam 2_0s-691s/frames/04424.png'
#
# img1 = Image.open(os.path.join(ROOT_DIR, imp1))
# img2 = Image.open(os.path.join(ROOT_DIR, imp2))

# POLY_CAM1_IN = [[240, 270], [410, 635], [480, 685], [705, 505], [550, 110]]
# POLY_CAM1_OUT = [[85, 350], [285, 755], [550, 730], [830, 500], [690, 35]]
# POLY_CAM2_IN = [[140, 0], [140, 200], [230, 235], [260, 185], [260, 0]]
# POLY_CAM2_OUT = [[80, 0], [80, 220], [240, 290], [330, 195], [330, 0]]

POLY_CAM1_IN = [[240, 270], [410, 650], [680, 450], [560, 100]]
POLY_CAM1_OUT = [[95, 330], [270, 750], [535, 760], [915, 480], [760, 0]]
POLY_CAM2_IN = [[140, 0], [140, 150], [260, 185], [260, 0]]
POLY_CAM2_OUT = [[80, 0], [80, 220], [240, 290], [330, 195], [330, 0]]

# frame_path = 'datasets/test 16_cam 1_0s-639s/frames'
# frames = sorted(os.listdir(os.path.join(ROOT_DIR, frame_path)))
GLOBAL_STEP = 0.1

true_bb_1 = load_data(pickle_path=os.path.join(ROOT_DIR, 'tests/true_bb_1.dict'))
true_bb_2 = load_data(pickle_path=os.path.join(ROOT_DIR, 'tests/true_bb_2.dict'))



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
                    print(3)
                last_track_seq['tr1'] = []
                last_track_seq['tr2'] = new_track_seq['tr2']
                count += 1
            else:
                if debug:
                    print(4)
                last_track_seq['tr2'] = new_track_seq['tr2']

        elif last_state == [False, True] and new_state == [True, False]:
            if min(new_track_seq['tr1']) - max(last_track_seq['tr2']) > MIN_EMPTY_SEQUENCE:
                if debug:
                    print(5)
                last_track_seq['tr2'] = []
                last_track_seq['tr1'] = new_track_seq['tr1']
                count += 1
            else:
                if debug:
                    print(6)
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

    def update_track_list(self, distance_limit: float, debug: bool = False):
        rel, deleted = [], []
        self.count_frames = []
        for i, trck in enumerate(self.track_list):
            # print(f" -- {i}, distance_limit={distance_limit}, trck={trck}")
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
            if self.frame_id - trck['frame_id'][-1] > MIN_EMPTY_SEQUENCE and \
                    (trck['check_in'][-1] or trck['check_out'][-1]):
                continue

            elif self.frame_id - trck['frame_id'][-1] > MIN_EMPTY_SEQUENCE and not trck['check_out'][-1]:
                if debug:
                    print(f"Check {self.name}", -(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1),
                              sum(trck['check_out'][:-(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]))
                deleted.append(i)
                self.count += 1

            # elif 1 < len(trck['boxes']) and trck['check_out'][-2:] == [True, False]:
            #     deleted.append(i)
            #     self.count += 1
            #     if debug:
            #         print(f"Check 2 {self.name}", -(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1),
            #                   sum(trck['check_out'][:-(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]))
            # elif len(trck['boxes']) > MIN_OBJ_SEQUENCE and \
            #         sum(trck['check_out'][:-(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]) == 0:
            #     if debug:
            #         print(f"Check2 {self.name}", -(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1),
            #               sum(trck['check_out'][:-(MIN_OBJ_SEQUENCE - self.frame_id + trck['frame_id'][-1] - 1)]))
            #     deleted.append(i)
            #     self.count += 1
            # elif 1 < len(trck['boxes']) <= MIN_OBJ_SEQUENCE and not trck['check_out'][-1]:
            #     if debug:
            #         print(f"Check3 {self.name}", [i['check_out'][-1] for i in self.track_list],
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
                    box = self.update_dead_boxes(frame_id=frame_id, new_box=box, distance_limit=speed_limit)

                if box:
                    self.current_boxes.append([box, check_in, check_out])

        # print("track", frame_id, [[int(c) for c in b[0][:4]] for b in self.current_boxes])

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
            dist_limit = 0.15 * diagonal
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
                f"track {self.name}, frame {frame_id}, count={self.count}\n"
                f"{[i['frame_id'] for i in self.track_list]}\n"
                f"{[i['boxes'] for i in self.track_list]}\n"
                f"{[i['check_out'] for i in self.track_list]}\n"
            )
        self.update_track_list(distance_limit=speed_limit, debug=debug)
        # print("track x2", self.count, [i['boxes'] for i in self.track_list], [i['check_out'] for i in self.track_list])


if __name__ == '__main__':
    # start, finish = (8 * 60 + 5) * 25, (8 * 60 + 13) * 25
    # start, finish = (7 * 60 + 50) * 25, (7 * 60 + 60) * 25
    start, finish = (0 * 60 + 0) * 25, (10 * 60 + 33) * 25
    tracker_1 = PolyTracker(polygon_in=POLY_CAM1_IN, polygon_out=POLY_CAM1_OUT, name='camera 1')
    tracker_2 = PolyTracker(polygon_in=POLY_CAM2_IN, polygon_out=POLY_CAM2_OUT, name='camera 2')
    names = ['carpet']
    colors = get_colors(names)
    out_size = (640, 360)
    out = cv2.VideoWriter(os.path.join(ROOT_DIR, 'temp/test.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 25,
                          (out_size[0], out_size[1] * 2))
    vc1 = cv2.VideoCapture()
    vc1.open(os.path.join(ROOT_DIR, 'videos/sync_test/test 16_cam 1_sync.mp4'))
    vc2 = cv2.VideoCapture()
    vc2.open(os.path.join(ROOT_DIR, 'videos/sync_test/test 16_cam 2_sync.mp4'))
    # print('fps =', vc1.get(cv2.CAP_PROP_FPS))

    # trfr1, trfr2 = [], []
    # trc = []
    count = 0
    last_track_seq = {'tr1': [], 'tr2': []}
    for i in range(0, finish):
        # itt = time.time()
        _, img1 = vc1.read()
        _, img2 = vc2.read()

        if i >= start:
            boxes_1 = true_bb_1[i]
            tracker_1.process(frame_id=i, boxes=boxes_1, img_shape=img1.shape[:2], debug=False)
            boxes_2 = true_bb_2[i]
            tracker_2.process(frame_id=i, boxes=boxes_2, img_shape=img2.shape[:2], debug=False)
            # print('================================================================')
            # print(f" - frame={i}, count={count}, boxes_1={[[int(c) for c in box[:4]] for box in boxes_1]},"
            #       f" boxes_2={[[int(c) for c in box[:4]] for box in boxes_2]}")

            count, last_track_seq = PolyTracker.combine_count(
                count=count,
                last_track_seq=last_track_seq,
                tracker_1_count_frames=copy.deepcopy(tracker_1.count_frames),
                tracker_2_count_frames=copy.deepcopy(tracker_2.count_frames),
                frame_id=i,
                debug=False
            )

            # Draw all figures on image
            img1 = PolyTracker.prepare_image(
                image=img1,
                colors=colors,
                tracker_current_boxes=tracker_1.current_boxes,
                polygon_in=POLY_CAM1_IN,
                polygon_out=POLY_CAM1_OUT,
                poly_width=5,
                # reshape=(img1.shape[1], img1.shape[0])
                reshape=out_size
            )
            # cv2.imshow('image', img1)
            # cv2.waitKey(0)
            img2 = PolyTracker.prepare_image(
                image=img2,
                colors=colors,
                tracker_current_boxes=tracker_2.current_boxes,
                polygon_in=POLY_CAM2_IN,
                polygon_out=POLY_CAM2_OUT,
                poly_width=2,
                reshape=out_size
            )

            img = np.concatenate((img1, img2), axis=0)
            headline = f"Обнаружено ковров: {count}\nТрекер 1: {tracker_1.count}\nТрекер 2: {tracker_2.count}"
            img = add_headline_to_cv_image(
                image=img,
                headline=headline
            )
            # cv_img = cv2.cvtColor(img)
            # cv2.imshow('image', img1)
            # cv2.waitKey(0)

            if (i + 1) % 100 == 0:
                logger.info(f"Frames {i + 1} / {finish} was processed. Current count: {count}")
            out.write(img)
            # break
    logger.info(f"\nFinal count={count}")
    out.release()
    # print("Total time:", time_converter(time.time() - st))
