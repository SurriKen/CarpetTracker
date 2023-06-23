import os
import cv2
import numpy as np

from parameters import ROOT_DIR

sync_videos = {
    'model_1': os.path.join(ROOT_DIR, 'videos/sync_test/test 30_cam 1_sync.mp4'),
    'model_2': os.path.join(ROOT_DIR, 'videos/sync_test/test 30_cam 2_sync.mp4'),
    # 'save_path': 'temp/test 22.mp4',
    'true_count': 0
}

vc1 = cv2.VideoCapture()
vc1.open(sync_videos.get("model_1"))
f1 = vc1.get(cv2.CAP_PROP_FRAME_COUNT)
fps1 = vc1.get(cv2.CAP_PROP_FPS)
w1 = int(vc1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(vc1.get(cv2.CAP_PROP_FRAME_HEIGHT))

vc2 = cv2.VideoCapture()
vc2.open(sync_videos.get("model_2"))
f2 = vc2.get(cv2.CAP_PROP_FRAME_COUNT)
fps2 = vc2.get(cv2.CAP_PROP_FPS)
w2 = int(vc2.get(cv2.CAP_PROP_FRAME_WIDTH))
h2 = int(vc2.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(fps1, fps2, f1, f2)
step = min([fps1, fps2])
range_1 = [(i, round(i * 1000 / fps1, 1)) for i in range(int(f1))]
range_2 = [(i, round(i * 1000 / fps2, 1)) for i in range(int(f2))]
(min_range, max_range) = (range_1, range_2) if step == fps1 else (range_2, range_1)
(min_vc, max_vc) = (vc1, vc2) if step == fps1 else (vc2, vc1)
print('range 1 =', range_1[:5])
print('range 2 =', range_2[:5])
print()


def get_closest_id(x: float, data: list[tuple, ...]) -> int:
    dist = [(abs(data[i][1] - x), i) for i in range(len(data))]
    dist = sorted(dist)
    # print('dist', x, data, dist)
    return dist[0][1]

f = f1 if step == fps1 else f2
for i in range(int(f)):
    _, frame1 = min_vc.read()
    # _, frame2 = max_vc.read()
    closest_id = get_closest_id(min_range[0][1], max_range[:5])
    closest = max_range[closest_id]
    # print(min_range[0], closest, closest_id)
    min_range.pop(0)
    ids = list(range(closest_id)) if closest_id else [0]
    ids = sorted(ids, reverse=True)
    # print(min_range[0], closest, closest_id, ids)
    for id in ids:
        max_range.pop(id)
        _, frame2 = max_vc.read()

    frame1 = cv2.resize(frame1, (640, 360))
    frame2 = cv2.resize(frame2, (640, 360))
    img = np.concatenate((frame1, frame2), axis=0)
    cv2.imshow(f'1', img)
    cv2.waitKey(100)
    if not min_range or not max_range:
        break
