import colorsys
import os
import pickle
import random


def save_dict(dict_, file_path, filename):
    with open(os.path.join(file_path, f"{filename}.dict"), 'wb') as handle:
        pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(pickle_path):
    with open(pickle_path, 'rb') as handle:
        b = pickle.load(handle)
    return b


def save_txt(txt, txt_path):
    with open(txt_path, 'w') as f:
        f.write(txt)


def load_txt(txt_path):
    with open(txt_path, 'r') as handle:
        b = handle.readlines()
    return b


def get_colors(name_classes: list):
    length = 10 * len(name_classes)
    hsv_tuples = [(x / length, 1., 1.) for x in range(length)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.shuffle(colors)
    return colors[:len(name_classes)]


if __name__ == '__main__':
    s = {'a': 25, 'b': 25.02, 'c': [146, 11], 'd': (1, 1, 1)}
    save_dict(s, 'bin', '111')
    xx = load_dict('bin/111.dict')
    for k, v in xx.items():
        print(k, v, type(v))
