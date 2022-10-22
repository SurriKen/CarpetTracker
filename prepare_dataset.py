import os
import random
import shutil
import time

import numpy as np
from PIL import Image
from utils import save_txt


class PrepareDataset:
    def __init__(self):
        pass

    @staticmethod
    def xml2terra_dataset(image_dir: str, xml_dir: str, dataset_name: str, save_path: str, shrink=False, limit=500):
        print("\nPreparing dataset for object detection...")
        tmp_folder = f"{save_path}/tmp"
        PrepareDataset.remove_empty_xml(xml_dir)
        try:
            os.mkdir(tmp_folder)
        except:
            shutil.rmtree(tmp_folder, ignore_errors=True)
            os.mkdir(tmp_folder)

        image_list, xml_list = [], []
        with os.scandir(image_dir) as folder:
            for f in folder:
                image_list.append(f.name)
        with os.scandir(xml_dir) as folder:
            for f in folder:
                xml_list.append(f.name)
        image_list_upd = []
        names_list = []
        coords_list = []
        frames = 0
        random.shuffle(image_list)
        for img in image_list:
            ext_length = len(img.split('.')[-1]) + 1
            if f"{img[:-ext_length]}.xml" in xml_list:
                if frames >= limit:
                    break
                image_list_upd.append(img)
                coord = PrepareDataset.read_xml(xml_path=f"{xml_dir}/{img[:-ext_length]}.xml", shrink=shrink)
                coords_list.append(coord)
                for i in coord['coords']:
                    if i[-1] not in names_list:
                        names_list.append(i[-1])
                frames += 1
            else:
                continue

        PrepareDataset.write_obj_data(num_classes=len(names_list), save_path=tmp_folder)
        names_list = sorted(names_list)
        PrepareDataset.write_obj_names(names=names_list, save_path=tmp_folder)
        image_list = image_list_upd
        PrepareDataset.write_train_txt(image_list=image_list, dataset_name=dataset_name, save_path=tmp_folder)
        os.mkdir(f"{tmp_folder}/Images")
        os.mkdir(f"{tmp_folder}/Annotation")
        for data in coords_list:
            img_name = data['filename']
            if shrink:
                image = Image.open(f"{image_dir}/{img_name}")
                new_image = image.resize((416, 416))
                new_image.save(f"{tmp_folder}/Images/{img_name}")
            else:
                shutil.copy2(f"{image_dir}/{img_name}", f"{tmp_folder}/Images/{img_name}")
            ext_length = len(img_name.split('.')[-1]) + 1
            txt_name = f"{img_name[:-ext_length]}.txt"
            coord_txt = ""
            for c in data['coords']:
                coord_txt = f"{coord_txt}\n{c[0]},{c[1]},{c[2]},{c[3]},{names_list.index(c[-1])}"
            save_txt(coord_txt[1:], f"{tmp_folder}/Annotation/{txt_name}")
        shutil.make_archive(f'{save_path}/{dataset_name}', 'zip', f"{save_path}/tmp")
        shutil.rmtree(tmp_folder, ignore_errors=True)
        print(f"Object detection dataset is ready! Dataset path: '{f'{save_path}/{dataset_name}'}'")

    @staticmethod
    def prepare_classificator_dataset(image_path: str, xml_path: str, dataset_name: str, save_path: str, limit=500):
        print("\nPreparing dataset for image classification...")
        if not image_path:
            print("No image_path")
            return None
        if not xml_path:
            print("No label_path")
            return None
        PrepareDataset.remove_empty_xml(xml_path)
        image_list, lbl_list = [], []
        with os.scandir(image_path) as folder:
            for f in folder:
                image_list.append(f.name)
        with os.scandir(xml_path) as folder:
            for f in folder:
                lbl_list.append(f.name)
        if not image_list:
            print("image_path is empty!")
        if not lbl_list:
            print("label_path is empty!")

        tmp_folder = f"{save_path}/tmp"
        try:
            os.mkdir(tmp_folder)
        except:
            shutil.rmtree(tmp_folder, ignore_errors=True)
            os.mkdir(tmp_folder)
        os.mkdir(f"{tmp_folder}/yes")
        os.mkdir(f"{tmp_folder}/no")

        labeled, not_labeled = 0, 0
        random.shuffle(image_list)
        if limit > len(image_list):
            length = len(image_list)
        else:
            length = limit
        st = time.time()
        for img in image_list:
            if (labeled + not_labeled + 1) % int(length * 0.05) == 0:
                print(f"{round((labeled + not_labeled + 1) * 100 / length, 0)}% complete...")
            name = img.split(".")[0]
            if f"{name}.xml" in lbl_list:
                # if labeled < limit:
                shutil.copy2(f"{image_path}/{img}", f"{tmp_folder}/yes/{img}")
                labeled += 1
                # else:
                #     continue
            else:
                # if not_labeled < limit:
                shutil.copy2(f"{image_path}/{img}", f"{tmp_folder}/no/{img}")
                not_labeled += 1
                # else:
                #     continue
            if labeled + not_labeled >= limit:
                break
        print(f"Prepare zip archive...")
        shutil.make_archive(f'{save_path}/{dataset_name}', 'zip', f"{tmp_folder}")
        shutil.rmtree(tmp_folder)
        print(f"Classificator dataset is ready! Images with boxes - {labeled}, images without boxes - {not_labeled}")
        print(f"Dataset path: '{f'{save_path}/{dataset_name}'}'. Preparation time={round(time.time() - st, 1)}s")

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
    def write_obj_data(num_classes, save_path):
        txt = f"classes = {num_classes}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/"
        save_txt(txt, f'{save_path}/obj.data')

    @staticmethod
    def write_obj_names(names, save_path):
        txt = ""
        for name in names:
            txt = f"{txt}\n{name}"
        txt = txt[1:]
        save_txt(txt, f'{save_path}/obj.names')

    @staticmethod
    def write_train_txt(image_list, dataset_name, save_path):
        txt = ""
        for name in image_list:
            txt = f"{txt}\n{dataset_name}/obj_train_data/{name}"
        txt = txt[1:]
        save_txt(txt, f'{save_path}/train.txt')

    @staticmethod
    def remove_empty_xml(xml_folder):
        xml_list = []
        with os.scandir(xml_folder) as fold:
            for f in fold:
                xml_list.append(f.name)
        for xml in xml_list:
            box_info = PrepareDataset.read_xml(f"{xml_folder}/{xml}")
            if not box_info['coords']:
                os.remove(f"{xml_folder}/{xml}")

    @staticmethod
    def image2array(image_path, target_size: tuple, scaler='no_scaler'):
        """
        :param target_size: (width, height)
        :param image_path: str
        :param scaler: str = no_scaler or min_max_scaler
        :return: np.ndarray
        """
        img = Image.open(image_path)
        (w, h) = img.size
        img = img.resize(target_size)
        if scaler == 'no_scaler':
            return np.expand_dims(np.array(img), 0), (w, h)
        else:
            return np.expand_dims(np.array(img), 0) / 255, (w, h)
        pass


if __name__ == "__main__":
    # image_dir = "E:/AI/CarpetTracker/init_frames/Train_0_300s/init_frames"
    # xml_dir = "E:/AI/CarpetTracker/init_frames/Train_0_300s/xml_labels"
    # save_path = "E:/AI/CarpetTracker/init_frames/Train_0_300s"
    image_dir = "E:/AI/CarpetTracker/init_frames/Air_1_24s/init_frames"
    xml_dir = "E:/AI/CarpetTracker/init_frames/Air_1_24s/xml_labels"
    save_path = "E:/AI/CarpetTracker/init_frames/Air_1_24s"

    PrepareDataset.xml2terra_dataset(
        image_dir=image_dir,
        xml_dir=xml_dir,
        dataset_name='air_tracker_yolo',
        save_path=save_path,
        shrink=True,
        limit=1000
    )
    # PrepareDataset.prepare_classificator_dataset(
    #     image_path=image_dir,
    #     xml_path=xml_dir,
    #     dataset_name="carpet_class2",
    #     save_path=save_path,
    #     limit=5000
    # )
