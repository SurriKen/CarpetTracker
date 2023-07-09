import os
from copy import deepcopy
import matplotlib.path as mpltPath
import cv2
import torch
import torchvision
from scipy import stats
import numpy as np
from torchvision.utils import draw_bounding_boxes

from parameters import MIN_OBJ_SEQUENCE, IMAGE_IRRELEVANT_SPACE_PERCENT, ROOT_DIR, MIN_EMPTY_SEQUENCE, \
    DEAD_LIMIT_PERCENT, GLOBAL_STEP


class Tracker:

    def __init__(self):
        self.obj_tracks = None
        self.coordinates, self.total_count, self.cluster_list, self.seq_count = [], [], [], []
        self.cur_obj, self.obj_count, self.ttc, self.step = 0, 0, 0, 0
        self.id_coords = []
        self.cluster_pred = {}
        self.tracked_id = []
        self.clust_coords = []
        self.coordinates_seq = []

    @staticmethod
    def get_object_count_and_coords(total_count: list, coords: list):
        res = [[]]
        clust_coords = [[]]
        cnt = 0
        seq_cnt = 0
        for item1, item2 in zip(total_count, total_count[1:]):  # pairwise iteration
            if item2 - item1 < MIN_OBJ_SEQUENCE:
                # The difference is 1, if we're at the beginning of a sequence add both
                # to the result, otherwise just the second one (the first one is already
                # included because of the previous iteration).
                if not res[-1]:  # index -1 means "last element".
                    res[-1].extend((item1, item2))
                    # print(clust_coords[-1])
                    clust_coords[-1].extend((coords[item1], coords[item2]))
                else:
                    res[-1].append(item2)
                    clust_coords[-1].append(coords[item2])
                    if len(res[-1]) >= MIN_OBJ_SEQUENCE:
                        r = [len(coords[x]) for x in res[-1][-MIN_OBJ_SEQUENCE:]]
                        if seq_cnt < int(np.average(r)):
                            cnt += int(np.average(r)) - seq_cnt
                            seq_cnt = int(np.average(r))
                        if seq_cnt > r[-1] and r[-1] == np.average(r):
                            seq_cnt = r[-1]
            elif res[-1]:
                # The difference isn't 1 so add a new empty list in case it just ended a sequence.
                res.append([])
                clust_coords.append([])
                seq_cnt = 0

        # In case "l" doesn't end with a "sequence" one needs to remove the trailing empty list.
        if not res[-1]:
            del res[-1]
        if not clust_coords[-1]:
            del clust_coords[-1]

        clust_coords_upd = deepcopy(clust_coords)
        for cl in clust_coords_upd:
            if len(cl) < MIN_OBJ_SEQUENCE:
                clust_coords.pop(clust_coords.index(cl))
        return cnt, clust_coords

    @staticmethod
    def get_object_count(count_1: list, count_2: list):
        total_count = []
        total_count.extend(count_1)
        total_count.extend(count_2)
        total_count = sorted(list(set(total_count)))

        res = [[]]
        res_upd = [[]]
        cnt = 0

        if total_count:
            for item1, item2 in zip(total_count, total_count[1:]):  # pairwise iteration
                if item2 - item1 < MIN_OBJ_SEQUENCE:
                    # The difference is 1, if we're at the beginning of a sequence add both
                    # to the result, otherwise just the second one (the first one is already
                    # included because of the previous iteration).
                    if not res[-1]:  # index -1 means "last element".
                        res[-1].extend((item1, item2))
                    else:
                        res[-1].append(item2)
                else:
                    res.append([])

            res_upd = []
            for cl in res:
                if len(cl) >= MIN_OBJ_SEQUENCE:
                    res_upd.append(cl)
                    cnt += 1

        return cnt, res_upd

    @staticmethod
    def get_distance(c1: list, c2: list):
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    @staticmethod
    def get_obj_box_squares(clust_coords):
        bbox = {}
        bb_emp_seq = {}
        for cluster in clust_coords:
            cur_len = 1
            cur_obj = []
            max_idx = 0
            keys = deepcopy(list(bbox.keys()))
            for k in keys:
                if k not in cur_obj and len(bbox[k]) < MIN_OBJ_SEQUENCE:
                    bbox.pop(k)
            if bbox:
                max_idx = max(list(bbox.keys()))
            for i, cl in enumerate(cluster):
                if i == 0:
                    cur_len = len(cl)
                    for k in range(cur_len):
                        bbox[k + max_idx + 1] = [cl[k]]
                        cur_obj.append(k + max_idx + 1)
                        bb_emp_seq[k + max_idx + 1] = 0
                else:
                    if cur_len == len(cl) and cur_len == 1:
                        # print('cur_len == len(cl) and cur_len == 1', i, cur_obj)
                        bbox[cur_obj[0]].append(cl[0])
                    elif cur_len == len(cl):
                        # print('cur_len == len(cl)', i, cur_obj)
                        box_i = [b for b in range(len(cl))]
                        for k in cur_obj:
                            lb_center = bbox[k][-1][1:3]
                            closest, dist = 0, 1000000
                            for idx in box_i:
                                x = Tracker.get_distance(lb_center, cl[idx][1:3])
                                if x < dist:
                                    dist = x
                                    closest = idx
                            box_i.pop(box_i.index(closest))
                            bbox[k].append(cl[closest])
                    elif cur_len > len(cl):
                        # print('cur_len > len(cl)', i, cur_obj)
                        box_i = [b for b in range(len(cl))]
                        box_i2 = [b for b in range(len(cl))]
                        cur_obj2 = deepcopy(cur_obj)
                        for b in box_i:
                            lb_center = cl[b][1:3]
                            closest, dist = 0, 1000000
                            for k in cur_obj2:
                                x = Tracker.get_distance(lb_center, bbox[k][-1][1:3])
                                if x < dist:
                                    dist = x
                                    closest = k
                            box_i2.pop(box_i2.index(b))
                            cur_obj2.pop(cur_obj2.index(closest))
                            bbox[closest].append(cl[b])
                            if not box_i2:
                                break
                        for k in cur_obj2:
                            cur_obj.pop(cur_obj.index(k))
                            bb_emp_seq[k] += 1
                            if bb_emp_seq[k] == MIN_OBJ_SEQUENCE:
                                cur_obj.pop(cur_obj.index(k))
                        cur_len = len(cl)
                    else:
                        # print('cur_len < len(cl)', i, cur_obj)
                        box_i = [b for b in range(len(cl))]
                        for k in cur_obj:
                            if bbox.get(k):
                                lb_center = bbox[k][-1][1:3]
                                closest, dist = 0, 1000000
                                for idx in box_i:
                                    x = Tracker.get_distance(lb_center, cl[idx][1:3])
                                    if x < dist:
                                        dist = x
                                        closest = idx
                                box_i.pop(box_i.index(closest))
                                bbox[k].append(cl[closest])
                        for idx in box_i:
                            cur_obj.append(max(cur_obj) + 1)
                            bbox[cur_obj[-1]] = [cl[idx]]
                            bb_emp_seq[cur_obj[-1]] = 0
                        cur_len = len(cl)

        sq = {}
        threshold = 3
        for k in bbox.keys():
            sq[k] = []
            for b in bbox[k]:
                sq[k].append(b[3] * b[4])
            z = np.abs(stats.zscore(sq[k]))
            out = np.where(z > threshold)[0]
            r = list(range(len(sq[k])))
            for idx in out:
                r.pop(r.index(idx))
            sq[k] = np.array(sq[k])[r]

        vecs = []
        for k in sq.keys():
            x = list(zip(sq[k], list(range(len(sq[k])))))
            x = sorted(x, reverse=True)
            x = x[:MIN_OBJ_SEQUENCE]
            x = [i[1] for i in x]
            # print(k, len(np.array(sq[k])[x]), np.mean(np.array(sq[k])[x]))
            vecs.append(np.array(sq[k])[x])
        return np.array(vecs)

    @staticmethod
    def get_obj_id(clust_coords, coords):
        coords_id = []
        bbox = {}
        bb_emp_seq = {}
        for cluster in clust_coords:
            cur_len = 1
            cur_obj = []
            max_idx = 0
            keys = deepcopy(list(bbox.keys()))
            for k in keys:
                if k not in cur_obj and len(bbox[k]) < MIN_OBJ_SEQUENCE:
                    bbox.pop(k)
            if bbox:
                max_idx = max(list(bbox.keys()))
            for i, cl in enumerate(cluster):
                if i == 0:
                    cur_len = len(cl)
                    for k in range(cur_len):
                        bbox[k + max_idx + 1] = [cl[k]]
                        cur_obj.append(k + max_idx + 1)
                        bb_emp_seq[k + max_idx + 1] = 0
                else:
                    # print('cur_obj', cur_obj)
                    if cur_len == len(cl) and cur_len == 1:
                        # print('cur_len == len(cl) and cur_len == 1', i, cur_obj, cur_len, len(cl))
                        bbox[cur_obj[0]].append(cl[0])
                    elif cur_len == len(cl):
                        # print('cur_len == len(cl)', i, cur_obj, cur_len, len(cl))
                        box_i = [b for b in range(len(cl))]
                        for k in cur_obj:
                            lb_center = bbox[k][-1][1:3]
                            closest, dist = 0, 1000000
                            for idx in box_i:
                                x = Tracker.get_distance(lb_center, cl[idx][1:3])
                                if x < dist:
                                    dist = x
                                    closest = idx
                            box_i.pop(box_i.index(closest))
                            bbox[k].append(cl[closest])
                    elif cur_len > len(cl):
                        # print('cur_len > len(cl)', i, cur_obj, cur_len, len(cl))
                        box_i = [b for b in range(len(cl))]
                        box_i2 = [b for b in range(len(cl))]
                        cur_obj2 = deepcopy(cur_obj)
                        for b in box_i:
                            lb_center = cl[b][1:3]
                            closest, dist = 0, 1000000
                            for k in cur_obj2:
                                x = Tracker.get_distance(lb_center, bbox[k][-1][1:3])
                                if x < dist:
                                    dist = x
                                    closest = k
                            box_i2.pop(box_i2.index(b))
                            cur_obj2.pop(cur_obj2.index(closest))
                            bbox[closest].append(cl[b])
                            if not box_i2:
                                break
                        for k in cur_obj2:
                            cur_obj.pop(cur_obj.index(k))
                            bb_emp_seq[k] += 1
                            if bb_emp_seq[k] == MIN_OBJ_SEQUENCE:
                                cur_obj.pop(cur_obj.index(k))
                        cur_len = len(cl)
                    else:
                        # print('cur_len < len(cl)', i, cur_obj, cur_len, len(cl))
                        box_i = [b for b in range(len(cl))]
                        for k in cur_obj:
                            if bbox.get(k):
                                lb_center = bbox[k][-1][1:3]
                                closest, dist = 0, 1000000
                                for idx in box_i:
                                    x = Tracker.get_distance(lb_center, cl[idx][1:3])
                                    if x < dist:
                                        dist = x
                                        closest = idx
                                box_i.pop(box_i.index(closest))
                                bbox[k].append(cl[closest])
                        for idx in box_i:
                            if cur_obj:
                                cur_obj.append(max(cur_obj) + 1)
                            else:
                                cur_obj.append(1)
                            bbox[cur_obj[-1]] = [cl[idx]]
                            bb_emp_seq[cur_obj[-1]] = 0
                        cur_len = len(cl)

        update_bbox = {}
        corr_box_keys = {}
        count = 1
        for k in sorted(list(bbox.keys())):
            if len(bbox.get(k)) >= MIN_OBJ_SEQUENCE:
                update_bbox[count] = bbox.get(k)
                corr_box_keys[k] = count
                count += 1

        for bb in coords:
            # print('bb', bb)
            for k in bbox.keys():
                if bb == bbox[k][-1] and k in corr_box_keys.keys():
                    coords_id.append(corr_box_keys.get(k))
                    break
                elif bb == bbox[k][-1]:
                    coords_id.append(0)

        return update_bbox, coords_id

    @staticmethod
    def put_box_on_image(save_path, results, labels, color_list, coordinates):
        image = results[0].orig_img[:, :, ::-1].copy()
        image = np.transpose(image, (2, 0, 1))
        w, h = image.shape[:2]
        image = torch.from_numpy(image)
        coord = []
        for box in coordinates[-1]:
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
    def remove_irrelevant_box(boxes: list, origin_shape: tuple):
        # bounding box in (ymin, xmin, ymax, xmax) format
        x_min = int(origin_shape[1] * IMAGE_IRRELEVANT_SPACE_PERCENT)
        x_max = int(origin_shape[1] - origin_shape[1] * IMAGE_IRRELEVANT_SPACE_PERCENT)
        y_min = int(origin_shape[0] * IMAGE_IRRELEVANT_SPACE_PERCENT)
        y_max = int(origin_shape[0] - origin_shape[0] * IMAGE_IRRELEVANT_SPACE_PERCENT)
        # new_boxes = []
        box_center = (int((boxes[0] + boxes[2]) / 2), int((boxes[1] + boxes[3]) / 2))
        # if boxes[0] < x_min or boxes[1] < y_min or boxes[2] > x_max or boxes[3] > y_max:
        #     return []
        # print(box_center, ((x_min, y_min), (x_max, y_max)))
        if box_center[0] not in range(x_min, x_max) or box_center[1] not in range(y_min, y_max):
            return []
        else:
            return boxes

    def process(self, predict, remove_perimeter_boxes=False):
        cur_coords = []
        # print(predict[0].im)
        if len(predict[0].boxes):
            # self.total_count.append(self.step)
            lines = []
            # lines_ = []
            # pred0 = [torch.clone(predict[0].orig_img)]
            for i, det0 in enumerate(predict[0].boxes):
                # print(det0)
                *xyxy0, conf0, cls0 = det0.boxes.tolist()[0]
                xyxy0 = [int(x) for x in xyxy0]
                if remove_perimeter_boxes:
                    xyxy0 = Tracker.remove_irrelevant_box(xyxy0, predict[0].orig_shape)
                if xyxy0:
                    xyxy0.extend([conf0, int(cls0)])
                    lines.append(xyxy0)

            self.total_count.append(self.step)
            self.coordinates.append(lines)
            cur_coords = lines
            # print(self.total_count, lines)
            self.seq_count.append(1)
            # self.coordinates_seq.append()

        else:
            self.coordinates.append([])
            self.seq_count.append(0)
            # self.coordinates_.append([])

        # print(self.coordinates[-1], self.coordinates_[-1])
        if self.ttc != len(self.total_count):
            # print(self.total_count)
            self.obj_count, self.clust_coords = Tracker.get_object_count_and_coords(self.total_count, self.coordinates)
            self.ttc = len(self.total_count)

        if cur_coords:
            # print('self.obj_count', self.obj_count)
            # print('clust_coords', clust_coords)
            # print('cur_coords', cur_coords)
            self.obj_tracks, id_coords = Tracker.get_obj_id(self.clust_coords, cur_coords)
            self.id_coords.append(id_coords)

        else:
            self.id_coords.append([])

        # print('id_coords', self.id_coords)
        # print('total_count', self.total_count)
        # print('cur_coords', cur_coords)
        # print('coordinates', self.coordinates[-1])
        # vecs = Tracker.get_obj_box_squares(clust_coords)
        # if vecs.any():
        #     # _, lbl_pred = kmeans_predict(
        #     #     model=Kmeans_model,
        #     #     lbl_dict=Kmeans_cluster_names,
        #     #     array=vecs[-1]
        #     # )
        #     if len(vecs) > self.cur_obj:
        #         self.cur_obj = len(vecs)
        #         self.cluster_list.append(lbl_pred)
        #     else:
        #         self.cluster_list[-1] = lbl_pred
        # cluster_pred = dict(collections.Counter(cluster_list))
        self.step += 1


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

