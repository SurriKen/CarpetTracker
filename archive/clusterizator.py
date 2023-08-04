import copy
import os
import pickle
import re
import shutil
from scipy import stats
import numpy as np
from sklearn.cluster import KMeans

from parameters import MIN_OBJ_SEQUENCE, CARPET_SIZE_LIST, NUM_CLUSTERS, KMEANS_MODEL_NAME
from utils import save_txt, load_txt, get_distance, save_data, load_data


class Clusterizator:

    def __init__(self):
        pass

    @staticmethod
    def read_xml(xml_path: str, shrink=False, new_width: int = 416, new_height: int = 416):
        with open(xml_path, 'r') as xml:
            lines = xml.readlines()
        xml = ''
        for l in lines:
            xml = f"{xml}{l}"
        filename = xml.split("<filename>")[1].split("</filename>")[0]
        size = xml.split("<size>")[1].split("</size>")[0]
        width = int(size.split("<width>")[1].split("</width>")[0])
        height = int(size.split("<height>")[1].split("</height>")[0])
        objects = xml.split('<object>')[1:]
        coords = []
        for obj in objects:
            name = obj.split("<name>")[1].split("</name>")[0]
            if shrink:
                xmin = int(int(obj.split("<xmin>")[1].split("</xmin>")[0]) / width * new_width)
                ymin = int(int(obj.split("<ymin>")[1].split("</ymin>")[0]) / height * new_height)
                xmax = int(int(obj.split("<xmax>")[1].split("</xmax>")[0]) / width * new_width)
                ymax = int(int(obj.split("<ymax>")[1].split("</ymax>")[0]) / height * new_height)
            else:
                xmin = int(obj.split("<xmin>")[1].split("</xmin>")[0])
                ymin = int(obj.split("<ymin>")[1].split("</ymin>")[0])
                xmax = int(obj.split("<xmax>")[1].split("</xmax>")[0])
                ymax = int(obj.split("<ymax>")[1].split("</ymax>")[0])
            coords.append([xmin, ymin, xmax, ymax, name])
        return {"width": width, "height": height, "coords": coords, "filename": filename}

    @staticmethod
    def remove_empty_xml(xml_folder):
        xml_list = []
        with os.scandir(xml_folder) as fold:
            for f in fold:
                xml_list.append(f.name)
        for xml in xml_list:
            box_info = Clusterizator.read_xml(f"{xml_folder}/{xml}")
            if not box_info['coords']:
                os.remove(f"{xml_folder}/{xml}")

    @staticmethod
    def xml2yolo_boxes(xml_path, total_length, save_path):
        print("Preparing boxes for yolov7 object detection...")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        else:
            shutil.rmtree(save_path, ignore_errors=True)
            os.mkdir(save_path)

        xml_list = []
        Clusterizator.remove_empty_xml(xml_path)
        with os.scandir(xml_path) as folder:
            for f in folder:
                xml_list.append(f"{f.name}")

        # print(sorted(xml_list))

        print("Preparing labels...")
        names_list = []
        for xml in xml_list:
            coord = Clusterizator.read_xml(
                xml_path=f"{xml_path}/{xml}",
                shrink=False
            )  # {"width": width, "height": height, "coords": coords, "filename": filename}
            for c in coord['coords']:
                if c[-1] not in names_list:
                    names_list.append(c[-1])
        names_list = sorted(names_list)
        print('names_list=', names_list)

        for i in range(total_length):
            if f"%05d.xml" % i in xml_list:
                txt = ""
                coord = Clusterizator.read_xml(
                    xml_path=f"{xml_path}/%05d.xml" % i,
                    shrink=False
                )  # {"width": width, "height": height, "coords": coords, "filename": filename}
                for c in coord['coords']:
                    xc = (c[2] + c[0]) / 2 / coord["width"]
                    yc = (c[3] + c[1]) / 2 / coord["height"]
                    w = (c[2] - c[0]) / coord["width"]
                    h = (c[3] - c[1]) / coord["height"]
                    txt = f"{txt}\n{names_list.index(c[-1])} {xc} {yc} {w} {h}"
                txt = txt[1:]
                save_txt(txt, f"{save_path}/{i}.txt")
            if (i + 1) % int(total_length * 0.1) == 0:
                print(f"{int((i + 1) * 100 / total_length)}% ({i + 1}/{total_length}) complete...")
        print(f"Object detection boxes for yolov7 is ready! Savet to path {save_path}")

    @staticmethod
    def prepare_kmean_data(pr_dir: list, save_path: str, total_length: int):
        max_count = 0
        for pr in pr_dir:
            try:
                os.mkdir(f"{pr}/v7_labels")
            except:
                shutil.rmtree(f"{pr}/v7_labels", ignore_errors=True)
                os.mkdir(f"{pr}/v7_labels")
            Clusterizator.xml2yolo_boxes(
                xml_path=f"{pr}/xml_labels",
                total_length=7500,
                save_path=f"{pr}/v7_labels"
            )

        try:
            os.mkdir(f"{save_path}")
        except:
            shutil.rmtree(f"{save_path}", ignore_errors=True)
            os.mkdir(f"{save_path}")

        for i, pr in enumerate(pr_dir):
            with os.scandir(f"{pr}/v7_labels") as fold:
                for f in fold:
                    id = int(f.name.split(".")[0])
                    shutil.copy2(f"{pr}/v7_labels/{f.name}", f"{save_path}/{id + total_length * i + 100}.txt")
                    if id + total_length * i + 100 > max_count:
                        max_count = id + total_length * i + 100
        return max_count

    @staticmethod
    def counter(box_folder, total_length):
        bb_list = []
        with os.scandir(box_folder) as folder:
            for f in folder:
                bb_list.append(f"{f.name}")

        coords, total_count = [], []
        for i in range(total_length):
            x = []
            if f"{i}.txt" in bb_list:
                x = load_txt(f"{box_folder}/{i}.txt")
                x_upd = []
                for j in x:
                    j = re.sub('\n', '', j)
                    j = j.split(' ')
                    j_upd = []
                    for jj in j:
                        j_upd.append(float(jj))
                    x_upd.append(j_upd)
                x = x_upd
            if len(x):
                total_count.append(i)
            coords.append(x)

        res = [[]]
        clust_coords = [[]]
        cnt = 0
        seq_cnt = 0
        for item1, item2 in zip(total_count, total_count[1:]):  # pairwise iteration
            if item2 - item1 < MIN_OBJ_SEQUENCE:
                # The difference is 1, if we're at the beginning of a sequence add both
                # to the result, otherwise just the second one (the first one is already
                # included because of the previous iteration).
                if not res[-1]:
                    res[-1].extend((item1, item2))
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
                res.append([])
                clust_coords.append([])
                seq_cnt = 0

        if not res[-1]:
            del res[-1]
        if not clust_coords[-1]:
            del clust_coords[-1]
        return cnt, res, clust_coords

    @staticmethod
    def distribute_coords(clust_coords):
        bbox = {}
        bb_emp_seq = {}
        for cluster in clust_coords:
            cur_len = 1
            cur_obj = []
            max_idx = 0
            keys = copy.deepcopy(list(bbox.keys()))
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
                                x = get_distance(lb_center, cl[idx][1:3])
                                if x < dist:
                                    dist = x
                                    closest = idx
                            box_i.pop(box_i.index(closest))
                            bbox[k].append(cl[closest])
                    elif cur_len > len(cl):
                        # print('cur_len > len(cl)', i, cur_obj)
                        box_i = [b for b in range(len(cl))]
                        box_i2 = [b for b in range(len(cl))]
                        cur_obj2 = copy.deepcopy(cur_obj)
                        for b in box_i:
                            lb_center = cl[b][1:3]
                            closest, dist = 0, 1000000
                            for k in cur_obj2:
                                x = get_distance(lb_center, bbox[k][-1][1:3])
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
                                    x = get_distance(lb_center, cl[idx][1:3])
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

        threshold = 3
        params = {}
        sq = {}
        for k in bbox.keys():
            params[k] = []
            sq[k] = []
            for b in bbox[k]:
                params[k].append([b[3], b[4]])
                sq[k].append(b[3] * b[4])
            z = np.abs(stats.zscore(sq[k]))
            out = np.where(z > threshold)[0]
            r = list(range(len(sq[k])))
            for idx in out:
                r.pop(r.index(idx))
            # print(k, out, len(params[k]))
            sq[k] = np.array(sq[k])[r]

        vecs = []
        for k in sq.keys():
            x = list(zip(sq[k], list(range(len(sq[k])))))
            x = sorted(x, reverse=True)
            x = x[:MIN_OBJ_SEQUENCE]
            x = [i[1] for i in x]
            # print(k, len(np.array(sq[k])[x]), np.mean(np.array(sq[k])[x]))
            vecs.append(np.array(sq[k])[x])
        vecs = np.array(vecs)
        return vecs

    @staticmethod
    def train_kmeans(array, num_clusters, save_path='', name='model'):
        print(f"train_kmeans on total {len(array)} objects")
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001)
        kmeans.fit(array)
        if save_path:
            Clusterizator.save_model(kmeans, save_path, name)
        stat = {}
        for l in range(num_clusters):
            stat[l] = []
        for i, v in enumerate(array):
            stat[kmeans.labels_[i]].append(np.mean(v))
        s = []
        for k, v in stat.items():
            s.append((np.mean(v), k))
        s = sorted(s, reverse=True)
        stat2 = {}
        for l, ss in enumerate(s):
            # print(l, ss, len(stat[ss[1]]))
            stat2[ss[1]] = CARPET_SIZE_LIST[l]
        if save_path:
            save_data(
                data=stat2,
                folder_path=save_path,
                filename=name
            )
        return kmeans, stat2

    @staticmethod
    def save_model(model, save_path, name='model'):
        with open(f"{save_path}/{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(path, dict_=False, name='model'):
        with open(f"{path}/{name}.pkl", "rb") as f:
            model = pickle.load(f)
        lbl_dict = {}
        if dict_:
            lbl_dict = load_data(pickle_path=f"{path}/{name}.dict")
        return model, lbl_dict

    @staticmethod
    def predict(kmeans_path, array):
        model, lbl_dict = Clusterizator.load_model(
            path=kmeans_path,
            dict_=True
        )
        pred = model.predict(array.reshape(1, -1))
        lbl = CARPET_SIZE_LIST[pred[0]]
        return pred, lbl

    @staticmethod
    def prepare_data_for_predict(box_list):
        sq = []
        for b in box_list:
            sq.append(b[3] * b[4])
        z = np.abs(stats.zscore(sq))
        out = np.where(z > 3)[0]
        r = list(range(len(sq)))
        for idx in out:
            r.pop(r.index(idx))
        sq = np.array(sq)[r]
        x = list(zip(sq, list(range(len(sq)))))
        x = sorted(x, reverse=True)
        x = x[:MIN_OBJ_SEQUENCE]
        x = [i[1] for i in x]
        return np.array(sq)[x]


if __name__ == "__main__":
    # pr_dir = [
    #     'datasets/Train_0_0s-300s',
    #     'datasets/Train_1_0s-300s',
    #     'datasets/Train_2_0s-300s',
    #     'datasets/Train_3_0s-300s',
    #     'datasets/Train_4_0s-300s',
    # ]
    # kmean_dataset = 'datasets/kmean_lbl'
    # # save_path = "datasets"
    # # max_count = Clusterizator.prepare_kmean_data(
    # #     pr_dir=pr_dir,
    # #     save_path='datasets/kmean_lbl',
    # #     total_length=7500,
    # # )
    # # print(max_count)
    # max_count = 37522
    # _, _, coord = Clusterizator.counter(
    #     box_folder=kmean_dataset,
    #     total_length=max_count
    # )
    # print(coord)
    # vec = Clusterizator.distribute_coords(coord)
    # kmeans, stat2 = Clusterizator.train_kmeans(
    #     array=vec,
    #     num_clusters=NUM_CLUSTERS,
    #     save_path='kmean_model',
    #     name=KMEANS_MODEL_NAME,
    # )
    # print(stat2)
    # train_kmeans on total 476 objects
    # 0 (0.046138166745805634, 3) 52
    # 1 (0.033061156639071984, 1) 118
    # 2 (0.021411180403265107, 2) 152
    # 3 (0.011910517626663461, 0) 154
    print([6, *[1, 2, 3]])

