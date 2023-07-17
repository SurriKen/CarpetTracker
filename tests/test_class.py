import copy
import os
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from parameters import *

csv_files = os.listdir(os.path.join(DATASET_DIR, 'videos/classification_videos/csv'))
stat = {}
for v in csv_files:
    csv = os.path.join(DATASET_DIR, 'videos/classification_videos/csv', v)
    obj = 0
    data = pd.read_csv(csv)
    try:
        carpet_size = list(data['Размер'])
    except:
        carpet_size = list(data['размер'])
    # print(carpet_size)
    ccc = dict(Counter(carpet_size))
    # print(v, ccc)
    for k in ccc.keys():
        if k in stat.keys():
            stat[k] += ccc[k]
        else:
            stat[k] = ccc[k]

# for k, v in stat.items():
#     print(k, '-', v)

# videos/classification_videos/csv/13-05 ВО.csv {'115*200': 57, '85*150': 56, '60*90': 7, '150*300': 33}
# videos/classification_videos/csv/16-10 ЦП.csv {'85*150': 57, '115*200': 60, '60*90': 8, '150*300': 27, '115*400': 5}
# videos/classification_videos/csv/МОС 19-40.csv {'60*90': 8, '115*200': 76, '150*300': 42, '85*150': 50, '115*400': 4}
# videos/classification_videos/csv/Ночь 20-11.csv {'115*200': 70, '85*150': 58, '150*300': 10, '60*90': 17}
# videos/classification_videos/csv/05.06.23.csv {'150*300': 43, '85*150': 45, '115*200': 47, '115*400': 2, '60*90': 6}
# videos/classification_videos/csv/05.06.23 вечер.csv {'60*90': 14, '115*200': 48, '150*300': 30, '85*150': 63, '115*400': 2}
# videos/classification_videos/csv/19.06.23 в 13.40.csv {'150*300': 39, '115*200': 64, '85*150': 45, '115*400': 1, '60*90': 10}
# videos/classification_videos/csv/20.06.23 в 14.02.csv {'115*200': 37, '85*150': 54, '60*90': 16, '150*300': 24, '115*400': 1}
# videos/classification_videos/csv/21.06.23 в 14.40.csv {'85*150': 35, '150*300': 37, '115*200': 65, '60*90': 6, '115*400': 10}
# videos/classification_videos/csv/21.06.23 в 16.44.csv {'85*150': 25, '115*200': 87, '60*90': 7, '150*300': 11}
# videos/classification_videos/csv/test 33_27.06 в 15.13.csv {'150*300': 36, '115*200': 33, '115*400': 2, '60*90': 4, '85*150': 68}
# videos/classification_videos/csv/test 34_26.06 в 16.48.csv {'85*150': 56, '115*200': 52, '150*300': 23, '115*400': 7, '60*90': 15}
# videos/classification_videos/csv/test 35_26.06 в 18.35.csv {'85*150': 69, '60*90': 18, '115*200': 40, '150*300': 31, '115*400': 8}
# videos/classification_videos/csv/test 36_27.06 в 13.49.csv {'115*200': 48, '85*150': 54, '150*300': 23, '60*90': 17, '115*400': 1}

csv_52 = pd.read_csv(os.path.join(DATASET_DIR, 'videos/classification_videos/csv/test 52.csv'))
try:
    carpet_size = list(csv_52['Размер'])
except:
    carpet_size = list(csv_52['размер'])
print(carpet_size)

pred = copy.deepcopy(carpet_size)
pred[10] = '60*90'
pred[61] = '85*150'
pred[63] = '115*200'
pred[78] = '115*200'
pred[79] = '60*90'
pred[90] = '115*200'
print(pred)


def get_confusion_matrix(y_true: list, y_pred: list, classes: list, save_path: str = '',
                         get_percent: bool = False) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print('cm\n', cm)
    if save_path:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        plt.savefig(save_path)
        plt.close()

    if get_percent:
        cm_percent = np.zeros_like(cm).astype('float')

        for i in range(len(cm)):
            total = np.sum(cm[i])
            for j in range(len(cm[i])):
                cm_percent[i][j] = round(cm[i][j] * 100 / total, 1)
        print('cm_percent\n', cm_percent)
        if save_path:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=classes)
            disp.plot()
            plt.savefig(f"{save_path[:-4]}_%.jpg")
            plt.close()
    return cm
print(sorted(list(set(carpet_size))))
cm = get_confusion_matrix(carpet_size, pred, classes=sorted(list(set(carpet_size))), get_percent=True)