import copy
import time

import numpy as np

from parameters import MIN_OBJ_SEQUENCE, MIN_EMPTY_SEQUENCE
from tracker import Tracker
from utils import time_converter, load_data

# tracker_1_dict = load_dict(pickle_path='/media/deny/Новый том/AI/CarpetTracker/tests/tracker_1_dict.dict')
# tracker_2_dict = load_dict(pickle_path='/media/deny/Новый том/AI/CarpetTracker/tests/tracker_2_dict.dict')
patterns = load_data(pickle_path='/media/deny/Новый том/AI/CarpetTracker/tests/patterns.dict')
true_bb_1 = load_data(pickle_path='/media/deny/Новый том/AI/CarpetTracker/tests/true_bb_1.dict')
true_bb_2 = load_data(pickle_path='/media/deny/Новый том/AI/CarpetTracker/tests/true_bb_2.dict')


def get_distribution(pattern, tracker_1_dict, tracker_2_dict) -> dict:
    # print("================================")
    # for k in tracker_1_dict.keys():
    #     print("get_distribution begin", k, tracker_1_dict[k]['frame_id'])
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
                        dist[i]['tr_num']
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
    # for k in tracker_1_dict.keys():
    #     print("get_distribution end", k, tracker_1_dict[k]['frame_id'])
    return dist


def get_center(coord: list[int, int, int, int]) -> tuple[int, int]:
    return int((coord[0] + coord[2]) / 2), int((coord[1] + coord[3]) / 2)


def tracker_analysis(pattern, main_track, main_cam):
    vecs = {}
    for id in pattern.get(main_cam):
        start = [int(i) for i in main_track.get(id).get('coords')[0][:4]]
        fin = [int(i) for i in main_track.get(id).get('coords')[-1][:4]]
        vector_x = get_center(fin)[0] - get_center(start)[0]
        vecs[id] = {'start': start, 'fin': fin, 'vector_x': vector_x,
                    'frame_start': main_track.get(id).get('frame_id')[0],
                    'frame_fin': main_track.get(id).get('frame_id')[-1]}

    for id in vecs.keys():
        vecs[id]['cross_vecs'] = {}
        vecs[id]['follow_id'] = []
        for id2 in vecs.keys():
            if id2 != id:
                vecs[id]['cross_vecs'][id2] = \
                    get_center(vecs[id2]['start'])[0] - get_center(vecs[id]['fin'])[0]
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


def combine_res(main_res, sec_res):
    # print('main_res', main_res)
    res = {
        main_res[id]['track_id']: {'seq_id': [], 'frame_idx': main_res[id]['frame_idx']} for id in range(len(main_res))}
    # print('RES', res)
    frame_dist = []
    for sid in range(len(sec_res)):
        for mid in range(len(main_res)):
            frame_dist.append(
                (abs(main_res[mid]['frame_idx'][0] - sec_res[sid]['frame_idx'][0]), mid, sid))
    frame_dist = sorted(frame_dist)
    # print("Frame dist", frame_dist)
    m_key = [i for i in range(len(main_res))]
    s_key = [i for i in range(len(sec_res))]
    # print('main_res end', main_res)
    for fr in frame_dist:
        if fr[1] in m_key and fr[2] in s_key:
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


def pattern_analisys(pattern: dict, tracker_1_dict: dict, tracker_2_dict: dict):
    if pattern.get('tr_num') == 1:
        return [pattern]
    else:
        if len(pattern.get('cam1')) > 1:
            res_1 = tracker_analysis(
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
        # print(f'res_1: {len(res_1)} {res_1}')

        if len(pattern.get('cam2')) > 1:
            res_2 = tracker_analysis(
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
        # print(f'res_2: {len(res_2)} {res_2}')

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
        # for k in tracker_1_dict.keys():
        #     print(f"pattern_analisys", k, tracker_1_dict[k]['frame_id'])
        res = combine_res(main_res, sec_res)
        # print('res', res)
        # for k in tracker_1_dict.keys():
        #     print(f"pattern_analisys", k, tracker_1_dict[k]['frame_id'])
        return [
            {'cam1': [id], 'cam2': res[id]['seq_id'], 'tr_num': 1,
             'sequence': res[id]['frame_idx']} for id in res.keys()
        ]


def update_pattern(pattern, tracker_1_dict, tracker_2_dict) -> list:
    # print("================================")
    # for k in tracker_1_dict.keys():
    #     print("update_pattern begin", k, tracker_1_dict[k]['frame_id'])
    # x = time.time()
    dist = get_distribution(
        pattern=pattern,
        tracker_1_dict=tracker_1_dict,
        tracker_2_dict=tracker_2_dict
    )
    # for k, v in dist.items():
    #     print(f"pat {k}: {v}")
    # for k in tracker_1_dict.keys():
    #     print("get_distribution end", k, tracker_1_dict[k]['frame_id'])
    # print('================================================================')
    # print("time Tracker.get_distribution:", len(pattern), time_converter(time.time() - x))
    # x = time.time()
    new_pat = []
    for i in dist.keys():
        pat = pattern_analisys(
            pattern=dist.get(i),
            tracker_1_dict=tracker_1_dict,
            tracker_2_dict=tracker_2_dict
        )
        # print(pat)
        pat = [p['sequence'] for p in pat]
        new_pat.extend(pat)
        # print("time Tracker.pattern_analisys:", len(dist), time_converter(time.time() - x))
        # for k in tracker_1_dict.keys():
        #     print(f"pattern_analisys {i}", k, tracker_1_dict[k]['frame_id'])
    # print('================================================================')
    return new_pat


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


# x = time.time()
# patterns = update_pattern(
#     pattern=patterns,
#     tracker_1_dict=tracker_1_dict,
#     tracker_2_dict=tracker_2_dict
# )
# print("time Tracker.update_pattern:", time_converter(time.time() - x))
start, finish = 400, 490
tracker_1 = Tracker()
tracker_2 = Tracker()
st = time.time()
cur_count = 0
old_patterns = []
for i in range(start, len(true_bb_1)):
    itt = time.time()
    # x = time.time()
    result = {'boxes': true_bb_1[i], 'orig_shape': (1080, 1920)}
    tracker_1.process(
        frame_id=i,
        predict=result,
        remove_perimeter_boxes=True,
        speed_limit_percent=0.01
    )
    result = {'boxes': true_bb_2[i], 'orig_shape': (360, 640)}
    tracker_2.process(
        frame_id=i,
        predict=result,
        remove_perimeter_boxes=False,
        speed_limit_percent=0.01
    )
    print("result", i,
          [[int(c) for c in box[:4]] for box in tracker_1.current_boxes],
          [[int(c) for c in box[:4]] for box in tracker_2.current_boxes])
    # print("time process:", time_converter(time.time() - x))

    # x = time.time()
    test_patterns = Tracker.get_pattern(
        input=Tracker.join_frame_id(
            tracker_dict_1=tracker_1.tracker_dict,
            tracker_dict_2=tracker_2.tracker_dict
        )
    )
    # print("time get_pattern:", time_converter(time.time() - x))

    # x = time.time()
    test_patterns = update_pattern(
        pattern=test_patterns,
        tracker_1_dict=tracker_1.tracker_dict,
        tracker_2_dict=tracker_2.tracker_dict
    )
    # print("time update_pattern:", time_converter(time.time() - x))

    # x = time.time()
    # if test_patterns:
    #     # print("test_patterns:", test_patterns)
    #     old_pattern_count, test_patterns, old_pat = clean_tracks(
    #         frame=i,
    #         pattern=test_patterns,
    #         tracker_1_dict=tracker_1.tracker_dict,
    #         tracker_2_dict=tracker_2.tracker_dict
    #     )
    #     cur_count += old_pattern_count
    #     if old_pat:
    #         old_patterns.extend(old_pat)
    # print("time clean_tracks:", time_converter(time.time() - x))
    tracker_1.current_id = []
    tracker_1.current_boxes = []
    tracker_2.current_id = []
    tracker_2.current_boxes = []
    print("Iteration:", i, "Iteration time:", time_converter(time.time() - itt), "Find patterns:", cur_count + len(test_patterns))
    print()
    if i > finish:
        print("Total time:", time_converter(time.time() - st))
        break


for i in range(len(old_patterns)):
    print('old_patterns', i, old_patterns[i])
print("===================================================")
for i in range(len(test_patterns)):
    print('test_patterns', i, test_patterns[i])
print("===================================================")
for k in tracker_1.tracker_dict.keys():
    print('tracker_1.tracker_dict', k, tracker_1.tracker_dict[k]['frame_id'])
#
# print("===================================================")
# for k in tracker_1_dict.keys():
#     print('tracker_1_dict.tracker_dict', k, tracker_1_dict[k]['frame_id'])

print("===================================================")
for k in tracker_2.tracker_dict.keys():
    print('tracker_2.tracker_dict', k, tracker_2.tracker_dict[k]['frame_id'])

# print("===================================================")
# for k in tracker_2_dict.keys():
#     print('tracker_2_dict', k, tracker_2_dict[k]['frame_id'])
#
print("===================================================")
for i, p in enumerate(patterns):
    if p[-1] > start and (p[-1] <= finish or finish in p):
        print('pattern', i, p)

# print("===================================================")
# for i, p in enumerate(test_patterns):
#     print('pattern', i, p)

# DOUBLE - 4:07
# pattern 60 [6214, 6215, 6216, 6217, 6218, 6219, 6220, 6221]
# pattern 61 [6236, 6237, 6238, 6239, 6240, 6241, 6242, 6243, 6244, 6245, 6246, 6247, 6248, 6249, 6250, 6251, 6252]

# EMPTY BETWEEN - 5:02
# pattern 70 [7494, 7495, 7496, 7497, 7498, 7499, 7508, 7509, 7510, 7511, 7512, 7513, 7514, 7515]
# pattern 71 [7712, 7713, 7714, 7715, 7716, 7717, 7718, 7719, 7720, 7725]

# DOUBLE - 5:30
# pattern 77 [8327, 8328, 8329, 8330, 8331, 8332, 8333, 8334, 8335, 8336, 8337, 8338, 8339, 8340, 8341, 8342, 8343]
# pattern 78 [8411, 8412, 8413, 8414, 8415, 8416, 8417, 8418, 8419, 8420, 8421, 8422, 8423, 8424, 8425, 8426,
# 8427, 8428, 8429, 8430, 8431, 8432]

# EMPTY BETWEEN = 6:46
# pattern 100 [10112, 10113, 10114, 10115, 10116, 10117, 10118, 10119, 10120, 10121, 10123, 10124,
# 10125, 10126, 10127, 10128, 10129, 10130, 10131, 10132, 10133, 10134, 10135, 10136, 10138, 10139, 10144, 10145,
# 10146, 10147, 10153, 10155, 10156, 10157, 10158, 10160, 10161, 10162, 10163, 10164, 10165, 10166, 10167, 10168,
# 10169, 10170, 10171, 10172, 10173, 10174, 10175, 10176, 10177, 10178, 10179, 10180, 10181, 10182, 10183]
# pattern 101 [10218, 10219, 10220, 10221, 10222, 10223, 10224, 10225, 10226, 10227, 10228, 10229, 10230, 10231, 10232,
# 10233, 10234, 10235, 10236, 10237, 10239]

# Test 15
#  71 - 4:21, 43 - 3:00