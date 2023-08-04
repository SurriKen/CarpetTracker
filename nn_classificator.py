import inspect
import os.path
import random
from collections import Counter
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from dataset_processing import DatasetProcessing, VideoClass
import time
from parameters import ROOT_DIR
from utils import logger, time_converter, plot_and_save_gragh, save_dict_to_table_txt, load_data, save_data, save_txt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class Net(nn.Module):
    def __init__(self, device='cpu', num_classes: int = 5, input_size=(12, 256, 256, 3),
                 frame_size=(128, 128), concat_axis: int = 2):
        super(Net, self).__init__()
        self.input_size = input_size
        self.conv3d_1 = nn.Conv3d(
            in_channels=input_size[-1], out_channels=16, kernel_size=3, padding='same', device=device)
        self.conv3d_2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding='same', device=device)
        self.conv3d_3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding='same', device=device)
        self.conv3d_4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding='same', device=device)
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

    def __init__(
            self, num_classes: int = 5, name: str = 'model', weights: str = None,
            device: str = 'cuda:0', input_size: tuple = (2, 256, 128, 1), frame_size: tuple[int, int] = (128, 128),
            concat_axis: int = 2):
        self.num_classes = num_classes
        self.device = device
        self.input_size = input_size
        self.frame_size = frame_size
        self.concat_axis = concat_axis
        self.torch_device = torch.device(device)
        self.weights = weights
        self.model = self.load_model(weights)
        self.history = {}
        try:
            os.mkdir(os.path.join(ROOT_DIR, 'video_class_train'))
        except:
            pass
        self.name = name

    def load_model(self, weights: str = '') -> nn.Module:
        if weights and weights.split('.')[-1] == 'pt':
            self.model = torch.jit.load(weights)
            self.input_size = self.model.input_size
            try:
                self.model.cuda(torch.device('cuda:0'))
            except:
                pass
        else:
            self.model = Net(device=self.device, num_classes=self.num_classes, input_size=self.input_size,
                             frame_size=self.frame_size, concat_axis=2)
        return self.model

    def save_model(self, name, mode: str = 'last'):
        model_scripted = torch.jit.script(self.model)  # Export to TorchScript
        model_scripted.save(os.path.join(ROOT_DIR, 'video_class_train', name, f"{mode}.pt"))  # Save

    # @staticmethod
    # def save_dataset(dataset: VideoClass, save_path: str):
    #     keys = list(dataset.__dict__.keys())
    #     array_keys = ['x_train', 'y_train', 'x_val', 'y_val']
    #     for k in keys:
    #         if k not in array_keys and type(getattr(dataset, k)) == np.ndarray:
    #             array_keys.append(k)
    #     print(array_keys)
    #     for k in array_keys:
    #         arr = np.array(getattr(dataset, k))
    #         print(arr.shape)
    #         np.save(os.path.join(save_path, f'{k}.npy'), arr)
    #         print(os.path.join(save_path, f'{k}.npy'))
    #     dict_ = {}
    #     for k in keys:
    #         if k not in array_keys:
    #             dict_[k] = getattr(dataset, k)
    #     print(dict_)
    #     save_data(dict_, save_path, 'dataset_data')

    # @staticmethod
    # def load_dataset(folder_path: str) -> VideoClass:
    #     dataset = VideoClass()
    #     array_keys = ['x_train', 'y_train', 'x_val', 'y_val']
    #     for k in array_keys:
    #         if os.path.isfile(os.path.join(folder_path, f"{k}.npy")):
    #             arr = np.load(os.path.join(folder_path, f"{k}.npy"), allow_pickle=True)
    #             setattr(dataset, k, arr)
    #     if os.path.isfile(os.path.join(folder_path, f"dataset_data.dict")):
    #         dict_ = load_data(os.path.join(folder_path, f"dataset_data.dict"))
    #         for k, v in dict_.items():
    #             setattr(dataset, k, v)
    #     return dataset

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
    def track_to_array(tracks: list[list, list], frame_size: tuple = (128, 128), num_frames: int = 16,
                       concat_axis: int = 2) -> np.ndarray:
        """
        tracks: a list of lists [camera_1, camera_2] where camera_X = [[frame_id_1, frame_id_2, ...],[box_1, box_2, ...]]
        """
        tr1 = tracks[0]
        tr2 = tracks[1]

        min_fr = min([min(tr1[0]), min(tr2[0])])
        max_fr = max([max(tr1[0]), max(tr2[0])])
        print(min_fr, max_fr)
        sequence = list(range(min_fr, max_fr + 1))
        seq_frame_1, seq_frame_2 = [], []
        for fr in range(len(sequence)):
            fr1 = np.zeros(frame_size)
            fr2 = np.zeros(frame_size)

            if fr in tr1[0]:
                if tr1[1][tr1[0].index(fr)]:
                    box1 = [int(bb * frame_size[i % 2]) for i, bb in enumerate(tr1[1][tr1[0].index(fr)])]
                    fr1[box1[1]:box1[3], box1[0]:box1[2]] = 1.
            fr1 = np.expand_dims(fr1, axis=-1)
            seq_frame_1.append(fr1)

            if fr in tr2[0]:
                if tr2[1][tr2[0].index(fr)]:
                    box2 = [int(bb * frame_size[i % 2]) for i, bb in enumerate(tr2[1][tr2[0].index(fr)])]
                    fr2[box2[1]:box2[3], box2[0]:box2[2]] = 1.
            fr2 = np.expand_dims(fr2, axis=-1)
            seq_frame_2.append(fr2)

        batch = np.array([[np.array(seq_frame_1), np.array(seq_frame_2)]])
        if num_frames:
            row = list(range(len(batch[0][0])))
            idx = VideoClassifier.resize_list(row, num_frames)
            batch = batch[:, :, idx, ...]
            if concat_axis in [1, 2, -1]:
                batch = np.concatenate([batch[:, 0, ...], batch[:, 1, ...]], axis=concat_axis)
            else:
                batch = np.concatenate([batch[:, 0, ...], batch[:, 1, ...]], axis=1)
                print("Concat_axis is our of range. Choose from None, 0, 1, 2 or -1. "
                      "Used default value concat_axis=None")
            # batch = torch.from_numpy(np.array(batch))
        return batch

    @staticmethod
    def create_box_video_dataset(
            box_path: str, split: float, frame_size: tuple = (128, 128)) -> VideoClass:
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
            box_path: str, split: float, frame_size: tuple = (128, 128)) -> VideoClass:
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
        # criterion = nn.L1Loss()
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

    def predict(self, array, model: nn.Module, classes: list = None) -> list:
        if classes is None:
            classes = []
        # if weights:
        #     self.load_model(weights)

        array = self.numpy_to_torch(array)
        output = model(array)
        output = output.cpu().detach().numpy() if self.device != 'cpu' else output.detach().numpy()
        print('predict output', np.argmax(output, axis=-1), classes, output)
        if classes:
            return [classes[i] for i in list(np.argmax(output, axis=-1))]
        return list(np.argmax(output, axis=-1))

    @staticmethod
    def evaluate_on_test_data(test_dataset: str, weights: str = '') -> np.ndarray:
        weights_folder = weights[:-len(weights.split('/')[-1])]
        save_cm = f"{weights_folder}Test_Confusion Matrix.jpg"
        test_loss = 0
        dataset = load_data(test_dataset)
        classes = sorted(list(dataset.keys()))
        vc = VideoClassifier(num_classes=len(classes), weights=weights)
        dataset = VideoClassifier.create_box_video_dataset(
            box_path=test_dataset,
            split=1.0,
            frame_size=vc.model.frame_size,
        )
        num_test_batches = len(dataset.x_train)

        # inp = [1, num_frames, *dataset.x_train[0][0][0].shape]
        # inp[concat_axis] = inp[concat_axis] * 2

        criterion = nn.CrossEntropyLoss()

        y_true, y_pred = [], []
        with torch.no_grad():
            for test_batch in range(num_test_batches):
                x_test = vc.get_x_batch(
                    x_train=dataset.x_train[test_batch: test_batch + 1], num_frames=vc.model.num_frames,
                    concat_axis=vc.model.concat_axis)
                y_test = vc.get_y_batch(
                    label=dataset.y_train[test_batch: test_batch + 1], num_labels=len(dataset.classes))
                y_true.append(dataset.classes[dataset.y_train[test_batch]])
                output = vc.model(x_test)
                y_pred.append(dataset.classes[np.argmax(output.cpu().detach().numpy(), axis=-1)[0]])
                loss = criterion(output, y_test)
                test_loss += loss.cpu().detach().numpy()

        cm = vc.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm, get_percent=True)
        print(f"Confusion Matrix were saved in folder '{weights_folder}'")
        return cm


if __name__ == "__main__":
    dataset_path = load_data(os.path.join(ROOT_DIR, 'tests/class_boxes_27_model5_full.dict'))
    classes = list(dataset_path.keys())
    cl = np.random.choice(classes)
    vid_list = list(dataset_path.get(cl).keys())
    vid = np.random.choice(vid_list)
    cameras = sorted(list(dataset_path.get(cl).get(vid).keys()))
    print(cl, vid, cameras[0])
    tr1 = dataset_path.get(cl).get(vid).get(cameras[0])
    tr2 = dataset_path.get(cl).get(vid).get(cameras[1])
    print(len(tr1), tr1)
    print(len(tr2), tr2)
    last_track_seq = {
        'tr1': [
            [187, 188, 189, 190, 191, 192, 193, 196, 199, 200, 201, 202, 203, 204],
            [[0.290625, 0.34814814814814815, 0.3697916666666667, 0.47314814814814815],
             [0.3020833333333333, 0.33240740740740743, 0.38333333333333336, 0.4861111111111111],
             [0.340625, 0.30092592592592593, 0.40520833333333334, 0.4648148148148148],
             [0.34010416666666665, 0.30092592592592593, 0.40520833333333334, 0.4648148148148148],
             [0.359375, 0.26944444444444443, 0.43020833333333336, 0.4388888888888889],
             [0.3723958333333333, 0.24351851851851852, 0.4401041666666667, 0.39166666666666666],
             [0.371875, 0.2175925925925926, 0.4609375, 0.337037037037037],
             [0.39166666666666666, 0.14907407407407408, 0.4864583333333333, 0.21666666666666667],
             [0.434375, 0.07685185185185185, 0.5401041666666667, 0.17777777777777778],
             [0.434375, 0.07592592592592592, 0.5395833333333333, 0.1787037037037037],
             [0.434375, 0.07685185185185185, 0.5395833333333333, 0.1787037037037037],
             [0.4395833333333333, 0.05740740740740741, 0.5328125, 0.16203703703703703],
             [0.4583333333333333, 0.05092592592592592, 0.5333333333333333, 0.16111111111111112],
             [0.46927083333333336, 0.05, 0.5333333333333333, 0.1648148148148148]]],
        'tr2': [
            [188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207],
            [[0.21875, 0.37222222222222223, 0.3234375, 0.4722222222222222],
             [0.2203125, 0.36666666666666664, 0.3359375, 0.5055555555555555],
             [0.2296875, 0.3277777777777778, 0.3828125, 0.5194444444444445],
             [0.24375, 0.3055555555555556, 0.3953125, 0.525],
             [0.2515625, 0.28888888888888886, 0.40625, 0.5138888888888888],
             [0.284375, 0.2722222222222222, 0.428125, 0.4861111111111111],
             [0.33125, 0.2861111111111111, 0.4734375, 0.4583333333333333],
             [0.3984375, 0.25555555555555554, 0.4953125, 0.4638888888888889],
             [0.446875, 0.24722222222222223, 0.540625, 0.4388888888888889],
             [0.478125, 0.2611111111111111, 0.5828125, 0.4083333333333333], [0.5, 0.275, 0.60625, 0.4],
             [0.534375, 0.29444444444444445, 0.653125, 0.45],
             [0.5578125, 0.3333333333333333, 0.7234375, 0.4638888888888889],
             [0.5921875, 0.35, 0.7515625, 0.49444444444444446],
             [0.6234375, 0.3472222222222222, 0.80625, 0.5305555555555556],
             [0.6421875, 0.35, 0.821875, 0.5611111111111111],
             [0.6828125, 0.35833333333333334, 0.846875, 0.6],
             [0.71875, 0.37222222222222223, 0.8703125, 0.6444444444444445],
             [0.74375, 0.4, 0.878125, 0.675],
             [0.7765625, 0.46111111111111114, 0.890625, 0.7083333333333334]]]}
    print(len(last_track_seq['tr1'][0]), last_track_seq['tr1'])
    print(len(last_track_seq['tr2'][0]), last_track_seq['tr2'])
    arr = VideoClassifier.track_to_array(
        tracks=[[list(range(len(tr1))), tr1], [list(range(len(tr2))), tr2]],
        # tracks=[last_track_seq['tr1'], last_track_seq['tr2']],
        frame_size=(128, 128), num_frames=12
    )
    # print(arr.shape)
    vc = VideoClassifier(num_classes=len(classes), frame_size=(128, 128), concat_axis=2,
                         weights=os.path.join(ROOT_DIR, 'video_class_train/model5_95%/last.pt'))
    # print(vc.input_size, vc.frame_size, vc.concat_axis)
    print(vc.predict(arr, classes=classes, model=vc.model))
    print()
    arr = VideoClassifier.track_to_array(
        # tracks=[[list(range(len(tr1))), tr1], [list(range(len(tr2))), tr2]],
        tracks=[last_track_seq['tr1'], last_track_seq['tr2']],
        frame_size=(128, 128), num_frames=12
    )
    model = torch.jit.load(os.path.join(ROOT_DIR, 'video_class_train/video data_94%/last.pt'))
    # print(arr.shape)
    print(vc.predict(arr, classes=classes, model=model))
    arr = vc.numpy_to_torch(arr)
    print(arr.shape)
    print(model(arr))
