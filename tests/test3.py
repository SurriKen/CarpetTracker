import matplotlib.pyplot as plt

import pandas as pd

from utils import save_txt

def save_dict_to_table_txt(data: dict, save_path: str):
    keys = data.keys()
    file = ''
    n = 0
    for k in keys:
        file = f'{file}{"{:<10} ".format(k)}'
        if len(data.get(k)) > n:
            n = len(data.get(k))
    file = f"{file}\n"

    for i in range(n):
        for k in data.keys():
            file = f'{file}{"{:<10} ".format(data.get(k)[i])}'
        file = f"{file}\n"
        # print("{:<10} {:<10} {:<10}".format(name, age, course))
    save_txt(txt=file[:-2], txt_path=save_path)


x = [1.523, 1.4404, 1.4124, 1.3995, 1.392]
y = [1.4456, 1.3924, 1.3688, 1.3553, 1.3472]
data = {
    'epoch': [0, 1, 2, 3, 4],
    'train_loss': x,
    'val_loss': y,
}
save_dict_to_table_txt(data, '1.txt')
# Epoch 1, train_loss= 1.523, val_loss = 1.4456, epoch time = 4 min 21 sec
# Epoch 2, train_loss= 1.4404, val_loss = 1.3924, epoch time = 4 min 28 sec
# Epoch 3, train_loss= 1.4124, val_loss = 1.3688, epoch time = 4 min 22 sec
# Epoch 4, train_loss= 1.3995, val_loss = 1.3553, epoch time = 4 min 19 sec
# Epoch 5, train_loss= 1.392, val_loss = 1.3472, epoch time = 4 min 18 sec