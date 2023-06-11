from parameters import MIN_EMPTY_SEQUENCE, MIN_OBJ_SEQUENCE

tracker_1_count_frames = [
    {'count': 1, 'frame': 1317, 'id': [1312, 1313, 1314, 1317, 1310, 1311]},
    {'count': 2, 'frame': 1372, 'id': [1369, 1370, 1371, 1372]},
    {'count': 3, 'frame': 1656, 'id': [1656, 1654, 1655]},
    {'count': 4, 'frame': 1730, 'id': [1728, 1729, 1730, 1726, 1727]},
]
tracker_2_count_frames = [
    {'count': 1, 'frame': 1321, 'id': [1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1311]},
    {'count': 2, 'frame': 1379, 'id': [1376, 1377, 1378, 1379, 1371, 1372, 1373, 1374, 1375]},
    {'count': 3, 'frame': 1487, 'id': [1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488]},
    {'count': 4, 'frame': 1616, 'id': [1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618]},
    {'count': 5, 'frame': 1669, 'id': [1664, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663]},
    {'count': 6, 'frame': 1736, 'id': [1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1727]},
]


def combine_count(count: int, last_track_seq: dict,
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

    if new_state == [False, False]:
        pass

    elif (last_state == [False, False] and new_state == [True, False]) or \
            (last_state == [True, False] and new_state == [True, False]):
        print(1)
        last_track_seq['tr1'] = new_track_seq['tr1']
        count += 1

    elif (last_state == [False, False] and new_state == [False, True]) or \
            (last_state == [False, True] and new_state == [False, True]):
        print(2)
        last_track_seq['tr2'] = new_track_seq['tr2']
        count += 1

    elif last_state == [True, False] and new_state == [False, True]:
        if min(new_track_seq['tr2']) - max(last_track_seq['tr1']) > MIN_OBJ_SEQUENCE:
            print(3)
            last_track_seq['tr1'] = []
            last_track_seq['tr2'] = new_track_seq['tr2']
            count += 1
        else:
            print(4)
            last_track_seq['tr2'] = new_track_seq['tr2']

    elif last_state == [False, True] and new_state == [True, False]:
        if min(new_track_seq['tr1']) - max(last_track_seq['tr2']) > MIN_EMPTY_SEQUENCE:
            print(5)
            last_track_seq['tr2'] = []
            last_track_seq['tr1'] = new_track_seq['tr1']
            count += 1
        else:
            print(6)
            last_track_seq['tr1'] = new_track_seq['tr1']

    elif last_state == [True, True] and new_state == [True, False]:
        if min(new_track_seq['tr1']) - max([max(last_track_seq['tr1']), max(last_track_seq['tr2'])]) > MIN_EMPTY_SEQUENCE:
            print(7)
            last_track_seq['tr2'] = []
            last_track_seq['tr1'] = new_track_seq['tr1']
            count += 1
        else:
            print(8)
            last_track_seq['tr1'] = new_track_seq['tr1']

    elif last_state == [True, True] and new_state == [False, True]:
        if min(new_track_seq['tr2']) - max([max(last_track_seq['tr1']), max(last_track_seq['tr2'])]) > MIN_EMPTY_SEQUENCE:
            print(9)
            last_track_seq['tr1'] = []
            last_track_seq['tr2'] = new_track_seq['tr2']
            count += 1
        else:
            print(10)
            last_track_seq['tr2'] = new_track_seq['tr2']

    else:
        print(11)
        last_track_seq['tr1'] = new_track_seq['tr1']
        last_track_seq['tr2'] = new_track_seq['tr2']
        count += 1

    return count, last_track_seq

count = 0
last_track_seq = {'tr1': [], 'tr2': []}
# state = [False, False]
for i in range(1300, 1750):
    tr1 = []
    for tr in tracker_1_count_frames:
        if i == tr['frame']:
            tr1 = tr['id']
            break
    tr2 = []
    for tr in tracker_2_count_frames:
        if i == tr['frame']:
            tr2 = tr['id']
            break

    count, last_track_seq = combine_count(
        count=count,
        last_track_seq=last_track_seq,
        tracker_1_count_frames=tr1,
        tracker_2_count_frames=tr2
    )

    print(f"frame {i}, count={count}, tr1={tr1}, tr2={tr2}")




