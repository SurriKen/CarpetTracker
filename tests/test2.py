
import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.path as mpltPath
from torchvision.utils import draw_bounding_boxes

from parameters import ROOT_DIR, MIN_EMPTY_SEQUENCE, MIN_OBJ_SEQUENCE
from utils import load_data, time_converter, get_colors, add_headline_to_cv_image, logger

imp1 = 'datasets/test 16_cam 1_0s-639s/frames/04424.png'
imp2 = 'datasets/test 16_cam 2_0s-691s/frames/04424.png'

img1 = Image.open(os.path.join(ROOT_DIR, imp1))
img2 = Image.open(os.path.join(ROOT_DIR, imp2))

POLY_CAM1_IN = [[185, 290], [360, 690], [590, 695], [820, 490], [665, 45]]
POLY_CAM1_OUT = [[95, 330], [270, 750], [645, 760], [915, 480], [760, 0]]
POLY_CAM2_IN = [[100, 0], [100, 215], [240, 285], [310, 200], [310, 0]]
POLY_CAM2_OUT = [[50, 0], [50, 225], [240, 340], [365, 240], [364, 0]]
frame_path = 'datasets/test 16_cam 1_0s-639s/frames'
# frames = sorted(os.listdir(os.path.join(ROOT_DIR, frame_path)))
GLOBAL_STEP = 0.1

true_bb_1 = load_data(pickle_path='/media/deny/Новый том/AI/CarpetTracker/tests/true_bb_1.dict')
true_bb_2 = load_data(pickle_path='/media/deny/Новый том/AI/CarpetTracker/tests/true_bb_2.dict')


class PolyTracker:
    def __init__(self, polygon_in: list, polygon_out: list):
        self.track_list = []
        self.count = 0
        self.max_id = 0
        self.current_boxes = []
        self.polygon_in = polygon_in
        self.polygon_out = polygon_out
        pass

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
        return dict(id=None, boxes=[], frame_id=[], check_in=[], check_out=[])

    @staticmethod
    def fill_track(track: dict, id: int, frame_id: int, box: list, check_in: bool, check_out: bool):
        track['id'] = id
        track['frame_id'].append(frame_id)
        track['boxes'].append(box)
        track['check_in'].append(check_in)
        track['check_out'].append(check_out)
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

    def process(self, frame_id: int, boxes: list, img_shape: tuple):
        # check if boxes are relevant
        self.current_boxes = []
        diagonal = ((img_shape[0]) ** 2 + (img_shape[1]) ** 2) ** 0.5
        step = GLOBAL_STEP * diagonal
        limit_in = self.expand_poly(self.polygon_in, -1 * step)
        limit_out = self.expand_poly(self.polygon_out, step)
        # print("================================")
        for box in boxes:
            center = self.get_center(box)
            # print(self.point_in_polygon(center, limit_in), self.point_in_polygon(center, limit_out))
            if not self.point_in_polygon(center, limit_in) and self.point_in_polygon(center, limit_out):
                check_in = self.point_in_polygon(center, self.polygon_in)
                check_out = self.point_in_polygon(center, self.polygon_out)
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
            # print('dist', dist, [i['boxes'][-1] for i in self.track_list], [i[0] for i in self.current_boxes])
            for d in dist:
                if tr_idxs and d[1] in tr_idxs and d[2] in box_idxs:
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
        rel = []
        # print()
        # print("track x1", self.count, [i['boxes'] for i in self.track_list], [i['check_out'] for i in self.track_list])
        for i, trck in enumerate(self.track_list):
            # print("track", i, trck.__dict__)
            if frame_id - trck['frame_id'][-1] > MIN_EMPTY_SEQUENCE:
                continue
            elif len(self.track_list[i]['boxes']) > MIN_OBJ_SEQUENCE and \
                    sum(trck['check_out'][:-(MIN_OBJ_SEQUENCE - frame_id + trck['frame_id'][-1] - 1)]) == 0:
                # print("Check", -(MIN_OBJ_SEQUENCE - frame_id + trck.frame_id[-1] - 1),
                #       sum(trck.check_out[:-(MIN_OBJ_SEQUENCE - frame_id + trck.frame_id[-1] - 1)]))
                self.count += 1
            elif len(self.track_list[i]['boxes']) < MIN_OBJ_SEQUENCE and not trck['check_out'][-1]:
                # print("Check2", [i.check_out[-1] for i in self.track_list], sum(trck.check_out[:-1]))
                self.count += 1
            else:
                rel.append(i)
        self.track_list = [self.track_list[i] for i in rel]
        # print("track x2", self.count, [i['boxes'] for i in self.track_list], [i['check_out'] for i in self.track_list])


start, finish = 50 * 25, 63 * 25
tracker_1 = PolyTracker(polygon_in=POLY_CAM1_IN, polygon_out=POLY_CAM1_OUT)
tracker_2 = PolyTracker(polygon_in=POLY_CAM2_IN, polygon_out=POLY_CAM2_OUT)
names = ['carpet']
colors = get_colors(names)
out_size = (640, 360)
out = cv2.VideoWriter(os.path.join(ROOT_DIR, 'temp/test.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 25, (out_size[0], out_size[1] * 2))
vc1 = cv2.VideoCapture()
vc1.open(os.path.join(ROOT_DIR, 'videos/sync_test/test 16_cam 1_sync.mp4'))
vc2 = cv2.VideoCapture()
vc2.open(os.path.join(ROOT_DIR, 'videos/sync_test/test 16_cam 2_sync.mp4'))
# print('fps =', vc1.get(cv2.CAP_PROP_FPS))

for i in range(0, finish):
    # itt = time.time()
    _, img1 = vc1.read()
    _, img2 = vc2.read()

    if i >= start:
        boxes_1 = true_bb_1[i]
        tracker_1.process(frame_id=i, boxes=boxes_1, img_shape=img1.shape[:2])
        boxes_2 = true_bb_2[i]
        tracker_2.process(frame_id=i, boxes=boxes_2, img_shape=img2.shape[:2])

        # Draw all figures on image
        img1 = PolyTracker.draw_polygons(polygons=POLY_CAM1_IN, image=img1, outline=(0, 255, 0), width=5)
        img1 = PolyTracker.draw_polygons(polygons=POLY_CAM1_OUT, image=img1, outline=(0, 0, 255), width=5)

        img2 = PolyTracker.draw_polygons(polygons=POLY_CAM2_IN, image=img2, outline=(0, 255, 0), width=2)
        img2 = PolyTracker.draw_polygons(polygons=POLY_CAM2_OUT, image=img2, outline=(0, 0, 255), width=2)

        labels1 = [f"carpet" for tr in tracker_1.current_boxes]
        current_boxes1 = [tr[0] for tr in tracker_1.current_boxes]
        if len(labels1) > 1:
            cl1 = colors * len(labels1)
        else:
            cl1 = colors

        labels2 = [f"carpet" for tr in tracker_2.current_boxes]
        current_boxes2 = [tr[0] for tr in tracker_2.current_boxes]
        if len(labels2) > 1:
            cl2 = colors * len(labels2)
        else:
            cl2 = colors

        img1 = PolyTracker.put_box_on_image(
            save_path=None,
            image=img1,
            labels=labels1,
            color_list=cl1,
            coordinates=current_boxes1,
        )
        for box in current_boxes1:
            img1 = cv2.circle(img1, PolyTracker.get_center([int(c) for c in box[:4]]), int(0.01 * img1.shape[0]), colors[0], -1)

        img2 = PolyTracker.put_box_on_image(
            save_path=None,
            image=img2,
            labels=labels2,
            color_list=cl2,
            coordinates=current_boxes2,
        )
        for box in current_boxes1:
            img2 = cv2.circle(img2, PolyTracker.get_center([int(c) for c in box[:4]]), int(0.01 * img2.shape[0]), colors[0], -1)

        img1 = np.array(img1)
        img1 = cv2.resize(img1, out_size)
        headline = f"Обнаружено ковров: {tracker_1.count}"
        img1 = add_headline_to_cv_image(
            image=img1,
            headline=headline
        )
        # print(img1.shape)

        img2 = np.array(img2)
        img2 = cv2.resize(img2, out_size)
        headline = f"Обнаружено ковров: {tracker_2.count}"
        img2 = add_headline_to_cv_image(
            image=img2,
            headline=headline
        )
        # print(img2.shape)

        img = np.concatenate((img1, img2), axis=0)
        # cv_img = cv2.cvtColor(img)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)

        if (i + 1) % 100 == 0:
            logger.info(f"Frames {i + 1} / {finish} was processed")
        out.write(img)
        # break
out.release()
# print(tracker_1.count)
# print("Total time:", time_converter(time.time() - st))
