from copy import deepcopy

import torch
import torchvision
from scipy import stats
import numpy as np
from torchvision.utils import draw_bounding_boxes

from parameters import MIN_OBJ_SEQUENCE, IMAGE_IRRELEVANT_SPACE_PERCENT


class Tracker:

    def __init__(self):
        self.obj_tracks = None
        self.coordinates, self.total_count, self.cluster_list = [], [], []
        self.coordinates_ = []
        self.cur_obj, self.obj_count, self.ttc, self.step = 0, 0, 0, 0
        self.id_coords = []
        self.cluster_pred = {}
        self.tracked_id = []

    @staticmethod
    def get_object_count(total_count: list, coords: list):
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
                    # print(clust_coords[-1], coords, item1, item2)
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

        for bb in coords:
            for k in bbox.keys():
                if bb == bbox[k][-1]:
                    coords_id.append(k)
                    break

        return bbox, coords_id

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

        else:
            self.coordinates.append([])
            self.coordinates_.append([])

        # print(self.coordinates[-1], self.coordinates_[-1])
        if self.ttc != len(self.total_count):
            self.obj_count, clust_coords = Tracker.get_object_count(self.total_count, self.coordinates)
            self.ttc = len(self.total_count)
            # print(self.obj_count, clust_coords)

        if cur_coords:
            # print(self.obj_count, clust_coords, cur_coords)
            self.obj_tracks, id_coords = Tracker.get_obj_id(clust_coords, cur_coords)
            self.id_coords.append(id_coords)

        else:
            self.id_coords.append([])

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
