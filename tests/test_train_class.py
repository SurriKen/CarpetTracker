import inspect
import os.path
import random
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from dataset_processing import DatasetProcessing, VideoClass
import time
from parameters import ROOT_DIR
from utils import logger, time_converter, plot_and_save_gragh, save_dict_to_table_txt, load_data, save_data, save_txt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

logger.info("\n    --- Running classif_models.py ---    \n")


class Net(nn.Module):
    def __init__(self, device='cpu', num_classes: int = 5, input_size=(12, 256, 256, 3)):
        super(Net, self).__init__()
        self.input_size = input_size
        self.conv3d_1 = nn.Conv3d(
            in_channels=input_size[-1], out_channels=32, kernel_size=3, padding='same', device=device)
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding='same', device=device)
        self.conv3d_3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding='same', device=device)
        self.conv3d_4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding='same', device=device)
        self.dense_3d = nn.Linear(in_features=128 * int(input_size[1] / 16) * int(input_size[2] / 16) * input_size[0],
                                  out_features=256, device=device)
        self.dense_3d_3 = nn.Linear(in_features=256, out_features=num_classes, device=device)
        self.post = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_1(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_2(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_3(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_4(x))), (1, 2, 2))
        x = x.reshape(x.size(0), -1)
        x = F.normalize(self.dense_3d(x))
        x = self.post(F.normalize(self.dense_3d_3(x)))
        return x


class VideoClassifier:

    def __init__(self, num_classes: int = 5, name: str = 'model', weights: str = None,
                 device: str = 'cuda:0', input_size: tuple = (2, 32)):
        self.num_classes = num_classes
        self.device = device
        self.input_size = input_size
        self.torch_device = torch.device(device)
        self.weights = weights
        self.model = None
        self.load_model(weights)
        self.history = {}
        try:
            os.mkdir(os.path.join(ROOT_DIR, 'video_class_train'))
        except:
            pass
        self.name = name

    def load_model(self, weights: str = '') -> None:
        if weights and weights.split('.')[-1] == 'pt':
            self.model = torch.jit.load(os.path.join(ROOT_DIR, weights))
        else:
            self.model = Net(device=self.device, num_classes=self.num_classes, input_size=self.input_size)

    def save_model(self, name, mode: str = 'last'):
        model_scripted = torch.jit.script(self.model)  # Export to TorchScript
        model_scripted.save(os.path.join(ROOT_DIR, 'video_class_train', name, f"{mode}.pt"))  # Save

    @staticmethod
    def save_dataset(dataset: VideoClass, save_path: str):
        keys = list(dataset.__dict__.keys())
        array_keys = ['x_train', 'y_train', 'x_val', 'y_val']
        for k in keys:
            if k not in array_keys and type(getattr(dataset, k)) == np.ndarray:
                array_keys.append(k)
        print(array_keys)
        for k in array_keys:
            arr = np.array(getattr(dataset, k))
            print(arr.shape)
            np.save(os.path.join(save_path, f'{k}.npy'), arr)
            print(os.path.join(save_path, f'{k}.npy'))
        dict_ = {}
        for k in keys:
            if k not in array_keys:
                dict_[k] = getattr(dataset, k)
        print(dict_)
        save_data(dict_, save_path, 'dataset_data')

    @staticmethod
    def load_dataset(folder_path: str) -> VideoClass:
        dataset = VideoClass()
        array_keys = ['x_train', 'y_train', 'x_val', 'y_val']
        for k in array_keys:
            if os.path.isfile(os.path.join(folder_path, f"{k}.npy")):
                arr = np.load(os.path.join(folder_path, f"{k}.npy"), allow_pickle=True)
                setattr(dataset, k, arr)
        if os.path.isfile(os.path.join(folder_path, f"dataset_data.dict")):
            dict_ = load_data(os.path.join(folder_path, f"dataset_data.dict"))
            for k, v in dict_.items():
                setattr(dataset, k, v)
        return dataset

    def get_x_batch(self, x_train: list, num_frames: int = None, concat_axis: int = None) -> torch.Tensor:
        if num_frames and 3 < num_frames:
            x1, x2 = [], []
            for batch in x_train:
                b1, b2 = batch[0], batch[1]
                sequence = list(range(len(b1)))
                idx = VideoClassifier.resize_list(sequence, num_frames)
                b1, b2 = b1[idx], b2[idx]
                x1.append(b1)
                x2.append(b2)
            if concat_axis in [1, 2, -1]:
                x_train = np.concatenate([x1, x2], axis=concat_axis)
            else:
                x_train = np.concatenate([x1, x2], axis=1)
                print("Concat_axis is our of range. Choose from None, 0, 1, 2 or -1. "
                      "Used default value concat_axis=None")
        x_train = torch.from_numpy(np.array(x_train))
        if 'cuda' in self.device:
            return x_train.to(self.torch_device, dtype=torch.float)
        else:
            return x_train.type(torch.FloatTensor)

    def numpy_to_torch(self, array: np.ndarray) -> torch.Tensor:
        array = torch.from_numpy(array)
        if 'cuda' in self.device:
            return array.to(self.torch_device, dtype=torch.float)
        else:
            return array

    def get_y_batch(self, label: list[int, ...], num_labels: int) -> torch.Tensor:
        lbl = DatasetProcessing.ohe_from_list(label, num_labels)
        if 'cuda' in self.device:
            return torch.tensor(lbl, dtype=torch.float, device=self.torch_device)
        else:
            return torch.tensor(lbl, dtype=torch.float)

    @staticmethod
    def get_confusion_matrix(y_true: list, y_pred: list, classes: list, save_path: str = '',
                             get_percent: bool = False) -> np.ndarray:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
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
            if save_path:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=classes)
                disp.plot()
                plt.savefig(f"{save_path[:-4]}_%.jpg")
                plt.close()
        return cm

    def fill_history(self, epoch: int = 0, train_loss: float = 0, val_loss: float = 0,
                     train_accuracy: float = 0, val_accuracy: float = 0, status: str = 'fill') -> None:
        if status == 'create':
            self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        elif status == 'fill':
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)

    @staticmethod
    def accuracy(cm_data: np.ndarray) -> float:
        acc = 0
        for i in range(len(cm_data)):
            acc += cm_data[i][i] / np.sum(cm_data[i])
        acc /= len(cm_data)
        return round(acc, 4)

    @staticmethod
    def create_class_dataset(x_path: str, y_path: str, split: float, classes: list,
                             shuffle: bool = False, remove_coords: bool = False) -> VideoClass:
        vc = VideoClass()
        vc.params['split'] = split
        vc.params['x_path'] = x_path
        vc.params['y_path'] = y_path
        classes = sorted(classes)
        vc.classes = classes
        data = np.load(x_path)
        lbl = np.load(y_path)
        if remove_coords:
            data = data[:, :, 11:]
        if shuffle:
            data = data.tolist()
            lbl = lbl.tolist()
            zip_data = list(zip(data, lbl))
            random.shuffle(zip_data)
            train, val = zip_data[:int(split * len(lbl))], zip_data[int(split * len(lbl)):]
            vc.x_train, vc.y_train = list(zip(*train))
            vc.x_val, vc.y_val = list(zip(*val))
            vc.x_train = np.array(vc.x_train)
            vc.y_train = np.array(vc.y_train)
            vc.x_val = np.array(vc.x_val)
            vc.y_val = np.array(vc.y_val)
        else:
            vc.x_train, vc.x_val = data[:int(split * len(data))], data[int(split * len(data)):]
            vc.y_train, vc.y_val = lbl[:int(split * len(lbl))], lbl[int(split * len(lbl)):]

        vc.train_stat = dict(Counter(vc.y_train.tolist()))
        vc.val_stat = dict(Counter(vc.y_val.tolist()))
        return vc

    @staticmethod
    def resize_list(sequence, length):
        if len(sequence) >= length:
            idx = list(range(len(sequence)))
            x2 = sorted(np.random.choice(idx, size=length, replace=False).tolist())
            y = [sequence[i] for i in x2]
        else:
            idx = list(range(len(sequence)))
            add = length - len(idx)
            idx.extend(np.random.choice(idx[1:-1], add))
            idx = sorted(idx)
            y = [sequence[i] for i in idx]
        return y

    @staticmethod
    def create_box_video_dataset(
            box_path: str, split: float, num_frames: int = 6, frame_size: tuple = (128, 128)) -> VideoClass:
        vc = VideoClass()
        vc.params['split'] = split
        vc.params['box_path'] = box_path
        dataset = load_data(box_path)
        vc.classes = sorted(list(dataset.keys()))

        data = []
        for class_ in dataset.keys():
            cl_id = vc.classes.index(class_)
            for vid in dataset[class_].keys():
                seq_frame_1, seq_frame_2 = [], []
                cameras = list(dataset[class_][vid].keys())
                if dataset[class_][vid] != {camera: [] for camera in cameras} and len(
                        dataset[class_][vid][cameras[0]]) > 2:
                    sequence = list(range(len(dataset[class_][vid][cameras[0]]))) if len(
                        dataset[class_][vid][cameras[0]]) \
                        else list(range(len(dataset[class_][vid][cameras[1]])))
                    # idx = VideoClassifier.resize_list(sequence, num_frames)
                    for fr in range(len(sequence)):
                        fr1 = np.zeros(frame_size)
                        fr2 = np.zeros(frame_size)

                        if dataset[class_][vid][cameras[0]][fr]:
                            box1 = [int(bb * frame_size[i % 2]) for i, bb in
                                    enumerate(dataset[class_][vid][cameras[0]][fr])]
                            fr1[box1[1]:box1[3], box1[0]:box1[2]] = 1.
                        fr1 = np.expand_dims(fr1, axis=-1)
                        seq_frame_1.append(fr1)

                        if dataset[class_][vid][cameras[1]][fr]:
                            box2 = [int(bb * frame_size[i % 2]) for i, bb in
                                    enumerate(dataset[class_][vid][cameras[1]][fr])]
                            fr2[box2[1]:box2[3], box2[0]:box2[2]] = 1.
                        fr2 = np.expand_dims(fr2, axis=-1)
                        seq_frame_2.append(fr2)

                    # seq_frame_1 = np.array(seq_frame_1)[idx]
                    # seq_frame_2 = np.array(seq_frame_2)[idx]
                    seq_frame_1 = np.array(seq_frame_1)
                    seq_frame_2 = np.array(seq_frame_2)
                    batch = [[seq_frame_1, seq_frame_2], cl_id]
                    # if concat_axis in [0, 1, 2, -1]:
                    #     batch = [np.concatenate([seq_frame_1, seq_frame_2], axis=concat_axis), cl_id]
                    # else:
                    #     print("Concat_axis is our of range. Choose from None, 0, 1, 2 or -1. "
                    #           "Used default value concat_axis=None")
                    data.append(batch)

        random.shuffle(data)
        x, y = list(zip(*data))
        # x = np.array(x)
        y = np.array(y)

        vc.x_train = x[:int(vc.params['split'] * len(x))]
        vc.y_train = y[:int(vc.params['split'] * len(y))]
        tr_stat = dict(Counter(vc.y_train))
        vc.train_stat = tr_stat
        vc.x_val = x[int(vc.params['split'] * len(x)):]
        vc.y_val = y[int(vc.params['split'] * len(y)):]
        v_stat = dict(Counter(vc.y_val))
        vc.val_stat = v_stat
        return vc

    @staticmethod
    def create_box_array_dataset(
            box_path: str, split: float, num_frames: int = 6, frame_size: tuple = (128, 128)) -> VideoClass:
        vc = VideoClass()
        vc.params['split'] = split
        vc.params['box_path'] = box_path
        dataset = load_data(box_path)
        vc.classes = sorted(list(dataset.keys()))

        data = []
        for class_ in dataset.keys():
            cl_id = vc.classes.index(class_)
            for vid in dataset[class_].keys():
                seq_frame_1, seq_frame_2 = [], []
                cameras = list(dataset[class_][vid].keys())
                if dataset[class_][vid] != {camera: [] for camera in cameras} and len(
                        dataset[class_][vid][cameras[0]]) > 2:
                    sequence = list(range(len(dataset[class_][vid][cameras[0]]))) if len(
                        dataset[class_][vid][cameras[0]]) \
                        else list(range(len(dataset[class_][vid][cameras[1]])))
                    # idx = VideoClassifier.resize_list(sequence, num_frames)
                    for fr in range(len(sequence)):
                        fr1 = np.zeros(frame_size)
                        fr2 = np.zeros(frame_size)

                        if dataset[class_][vid][cameras[0]][fr]:
                            box1 = [int(bb * frame_size[i % 2]) for i, bb in
                                    enumerate(dataset[class_][vid][cameras[0]][fr])]
                            fr1[box1[1]:box1[3], box1[0]:box1[2]] = 1.
                        fr1 = np.expand_dims(fr1, axis=-1)
                        seq_frame_1.append(fr1)

                        if dataset[class_][vid][cameras[1]][fr]:
                            box2 = [int(bb * frame_size[i % 2]) for i, bb in
                                    enumerate(dataset[class_][vid][cameras[1]][fr])]
                            fr2[box2[1]:box2[3], box2[0]:box2[2]] = 1.
                        fr2 = np.expand_dims(fr2, axis=-1)
                        seq_frame_2.append(fr2)

                    # seq_frame_1 = np.array(seq_frame_1)[idx]
                    # seq_frame_2 = np.array(seq_frame_2)[idx]
                    seq_frame_1 = np.array(seq_frame_1)
                    seq_frame_2 = np.array(seq_frame_2)
                    batch = [[seq_frame_1, seq_frame_2], cl_id]
                    # if concat_axis in [0, 1, 2, -1]:
                    #     batch = [np.concatenate([seq_frame_1, seq_frame_2], axis=concat_axis), cl_id]
                    # else:
                    #     print("Concat_axis is our of range. Choose from None, 0, 1, 2 or -1. "
                    #           "Used default value concat_axis=None")
                    data.append(batch)

        random.shuffle(data)
        x, y = list(zip(*data))
        # x = np.array(x)
        y = np.array(y)

        vc.x_train = x[:int(vc.params['split'] * len(x))]
        vc.y_train = y[:int(vc.params['split'] * len(y))]
        tr_stat = dict(Counter(vc.y_train))
        vc.train_stat = tr_stat
        vc.x_val = x[int(vc.params['split'] * len(x)):]
        vc.y_val = y[int(vc.params['split'] * len(y)):]
        v_stat = dict(Counter(vc.y_val))
        vc.val_stat = v_stat
        return vc

    def train(self, dataset: VideoClass, epochs: int, batch_size: int = 1, weights: str = '',
              lr: float = 0.005, num_frames: int = 6, concat_axis: int = 2, save_dataset: bool = True,
              load_dataset_path: str = '') -> None:
        # try:
        if weights:
            self.load_model(weights)
        stop = False
        i = 1
        name = f"{self.name}_{i}"
        try:
            os.mkdir(os.path.join(ROOT_DIR, 'video_class_train'))
        except:
            pass

        while not stop:
            if name in os.listdir(os.path.join(ROOT_DIR, 'video_class_train')):
                i += 1
                name = f"{self.name}_{i}"
            else:
                os.mkdir(os.path.join(ROOT_DIR, 'video_class_train', name))
                stop = True
        print(os.path.join(ROOT_DIR, 'video_class_train', name))
        if load_dataset_path:
            dataset = self.load_dataset(load_dataset_path)

        if save_dataset:
            self.save_dataset(dataset, os.path.join(ROOT_DIR, 'video_class_train', name))

        st = time.time()
        logger.info("Training is started\n")
        num_classes = len(dataset.classes)
        num_train_batches = int(len(dataset.x_train) / batch_size)
        train_seq = list(np.arange(len(dataset.x_train)))
        num_val_batches = len(dataset.x_val)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_loss, best_acc = 10000., 0.

        logger_batch_markers = []
        for i in range(10):
            logger_batch_markers.append(int(num_train_batches * (i + 1) / 10))

        txt = f"training parameters:\n" \
              f"- name: {name},\n" \
              f"- save path: {os.path.join(ROOT_DIR, 'video_class_train', name)}\n" \
              f"- optimizer: {optimizer.__dict__.get('_zero_grad_profile_name')}\n" \
              f"- optimizr params: {optimizer.state_dict().get('param_groups')[0]}\n" \
              f"\n- Model structure:\n" \
              f"{inspect.getsource(self.model.__init__) if not weights and not self.weights else ''}\n" \
              f"{inspect.getsource(self.model.forward) if not weights and not self.weights else ''}\n"
        logger.info(txt)
        save_txt(txt, os.path.join(ROOT_DIR, 'video_class_train', name, f"model_info.txt"))

        train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []
        self.fill_history(status='create')
        for epoch in range(epochs):
            st_ep = time.time()
            random.shuffle(train_seq)
            train_loss, train_acc = 0., 0.
            y_true, y_pred = [], []
            for batch in range(num_train_batches):
                # x_batch = dataset.x_train[train_seq[batch * batch_size:(batch + 1) * batch_size]]
                x_batch = [dataset.x_train[i] for i in train_seq[batch * batch_size:(batch + 1) * batch_size]]
                x_train = self.get_x_batch(x_train=x_batch, num_frames=num_frames, concat_axis=concat_axis)
                y_batch = dataset.y_train[train_seq[batch * batch_size:(batch + 1) * batch_size]]
                y_train = self.get_y_batch(label=y_batch, num_labels=num_classes)
                y_true.extend([dataset.classes[i] for i in y_batch])
                output = self.model(x_train)
                y_pred.extend([dataset.classes[i] for i in np.argmax(output.cpu().detach().numpy(), axis=-1)])
                loss = criterion(output, y_train)
                train_loss += loss.cpu().detach().numpy()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                if batch + 1 in logger_batch_markers:
                    save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, f'Train_Confusion Matrix.jpg')
                    cm = self.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm, get_percent=True)
                    train_acc = self.accuracy(cm)
                    logger.info(
                        f"  -- Epoch {epoch + 1}, batch {batch + 1} / {num_train_batches}, "
                        f"train_loss= {round(train_loss / (batch + 1), 4)}, "
                        f"train_accuracy= {train_acc}, "
                        f"average batch time = {round((time.time() - st_ep) * 1000 / (batch + 1), 1)} ms, "
                        f"time passed = {time_converter(int(time.time() - st))}"
                    )

            save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, f'Train_Confusion Matrix.jpg')
            cm = self.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm, get_percent=True)
            train_acc = self.accuracy(cm)
            train_loss_hist.append(round(train_loss / num_train_batches, 4))
            train_acc_hist.append(train_acc)

            val_loss = 0
            y_true, y_pred = [], []

            with torch.no_grad():
                for val_batch in range(num_val_batches):
                    x_val = self.get_x_batch(x_train=dataset.x_val[val_batch: val_batch + 1],
                                             num_frames=num_frames, concat_axis=concat_axis)
                    y_val = self.get_y_batch(label=dataset.y_val[val_batch: val_batch + 1],
                                             num_labels=num_classes)
                    y_true.append(dataset.classes[dataset.y_val[val_batch]])
                    output = self.model(x_val)
                    y_pred.append(dataset.classes[np.argmax(output.cpu().detach().numpy(), axis=-1)[0]])
                    loss = criterion(output, y_val)
                    val_loss += loss.cpu().detach().numpy()

            save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, f'Val_Confusion Matrix.jpg')
            cm = self.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm, get_percent=True)
            val_acc = self.accuracy(cm)
            val_loss_hist.append(round(val_loss / num_val_batches, 4))
            val_acc_hist.append(val_acc)

            plot_and_save_gragh(train_loss_hist, 'epochs', 'Loss', 'Train loss',
                                os.path.join(ROOT_DIR, 'video_class_train', name))
            plot_and_save_gragh(val_loss_hist, 'epochs', 'Loss', 'Val loss',
                                os.path.join(ROOT_DIR, 'video_class_train', name))
            plot_and_save_gragh(train_acc_hist, 'epochs', 'Accuracy', 'Train Accuracy',
                                os.path.join(ROOT_DIR, 'video_class_train', name))
            plot_and_save_gragh(val_acc_hist, 'epochs', 'Accuracy', 'Val Accuracy',
                                os.path.join(ROOT_DIR, 'video_class_train', name))
            self.fill_history(
                epoch=epoch,
                train_loss=round(train_loss / num_train_batches, 4),
                val_loss=round(val_loss / num_val_batches, 4),
                train_accuracy=train_acc,
                val_accuracy=val_acc,
            )
            save_dict_to_table_txt(
                self.history, os.path.join(ROOT_DIR, 'video_class_train', name, 'train_history.txt'))

            self.save_model(name=name, mode='last')
            if val_acc >= best_acc:
                self.save_model(name=name, mode='best')
                best_acc = val_acc
                logger.info('\nBest weights were saved')

            logger.info(f"\nEpoch {epoch + 1}, train_loss= {round(train_loss / num_train_batches, 4)}, "
                        f"val_loss = {round(val_loss / num_val_batches, 4)}, "
                        f"train_accuracy= {train_acc}, val_accuracy = {val_acc}, "
                        f"epoch time = {time_converter(int(time.time() - st_ep))}\n")

        logger.info(f"Training is finished, "
                    f"train time = {time_converter(int(time.time() - st))}\n")

    def predict(self, array, weights: str = '', classes: list = None) -> list:
        if classes is None:
            classes = []
        if weights:
            self.load_model(weights)
        array = self.numpy_to_torch(array)
        output = self.model(array)
        output = output.cpu().detach().numpy() if self.device != 'cpu' else output.detach().numpy()
        if classes:
            return [classes[i] for i in list(np.argmax(output, axis=-1))]
        return list(np.argmax(output, axis=-1))


if __name__ == "__main__":
    # x = np.load(os.path.join(ROOT_DIR, 'tests/x_train_stat.npy'))
    # y = np.load(os.path.join(ROOT_DIR, 'tests/y_train_stat.npy'))
    classes = ['115x200', '115x400', '150x300', '60x90', '85x150']
    st = time.time()
    device = 'cuda:0'
    num_frames = 16
    concat_axis = 2
    name = 'video data'
    dataset_path = ''
    # device = 'cpu'
    dataset = VideoClassifier.create_box_video_dataset(
        box_path=os.path.join(ROOT_DIR, 'tests/class_boxes_26_model3_full.dict'),
        split=0.9,
        frame_size=(128, 128),
    )

    logger.info(f'Dataset was formed\nclasses {dataset.classes}\ntrain_stat {dataset.train_stat}\n'
                f'val_stat {dataset.val_stat}\nparameters: {dataset.params}\n')

    inp = [1, num_frames, *dataset.x_val[0][0][0].shape]
    inp[concat_axis] = inp[concat_axis] * 2
    print('input size', inp)
    vc = VideoClassifier(num_classes=len(dataset.classes), weights='',
                         input_size=tuple(inp[1:]), name=name, device=device)
    print("Training is started")
    vc.train(
        dataset=dataset,
        epochs=100,
        batch_size=4,
        lr=0.00002,
        num_frames=num_frames,
        concat_axis=concat_axis,
        save_dataset=False,
        load_dataset_path=''
    )
