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
    def xml2terra_dataset(project_paths: list, dataset_name: str, save_path: str, shrink=False, limit=500):
        print("\nPreparing dataset for object detection...")
        tmp_folder = f"{save_path}/tmp"
        # PrepareDataset.remove_empty_xml(xml_dir)
        try:
            os.mkdir(tmp_folder)
        except:
            shutil.rmtree(tmp_folder, ignore_errors=True)
            os.mkdir(tmp_folder)

        image_list, xml_list = [], []
        for pr in project_paths:
            PrepareDataset.remove_empty_xml(f"{pr}/xml_labels")
            with os.scandir(f"{pr}/frames") as folder:
                for f in folder:
                    image_list.append((f"{pr}", f"{f.name}"))
            with os.scandir(f"{pr}/xml_labels") as folder:
                for f in folder:
                    xml_list.append((f"{pr}", f"{f.name}"))
        image_list_upd = []
        names_list = []
        coords_list = []
        frames = 0
        random.shuffle(image_list)
        print("Preparing images and labels...")
        for img in image_list:
            ext_length = len(img[1].split('.')[-1]) + 1
            if (img[0], f"{img[1][:-ext_length]}.xml") in xml_list:
                if frames >= limit:
                    break
                image_list_upd.append(img)
                coord = PrepareDataset.read_xml(
                    xml_path=f"{img[0]}/xml_labels/{img[1][:-ext_length]}.xml",
                    shrink=shrink
                )
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
        print("Sorting and saving dataset...")

        for i, data in enumerate(coords_list):
            if (i + 1) % int(len(coords_list) * 0.05) == 0:
                print(f"{int((i + 1) * 100 / len(coords_list))}% complete...")
            # img_name = data['filename']
            pr_path, img_name = image_list[i]
            # pr_name = pr_path.split("/")[-1]
            if shrink:
                image = Image.open(f"{pr_path}/frames/{img_name}")
                new_image = image.resize((416, 416))
                new_image.save(f"{tmp_folder}/Images/{i}.png")
            else:
                shutil.copy2(f"{pr_path}/frames/{img_name}", f"{tmp_folder}/Images/{i}.png")
            # ext_length = len(img_name.split('.')[-1]) + 1
            txt_name = f"{i}.txt"
            coord_txt = ""
            for c in data['coords']:
                coord_txt = f"{coord_txt}\n{c[0]},{c[1]},{c[2]},{c[3]},{names_list.index(c[-1])}"
            save_txt(coord_txt[1:], f"{tmp_folder}/Annotation/{txt_name}")
        print(f"Prepare zip archive...")
        shutil.make_archive(f'{save_path}/{dataset_name}', 'zip', f"{save_path}/tmp")
        shutil.rmtree(tmp_folder, ignore_errors=True)
        print(f"Object detection dataset is ready! Dataset path: '{f'{save_path}/{dataset_name}'}'")

    @staticmethod
    def prepare_classificator_dataset(project_paths: list, dataset_name: str, save_path: str, resize=(0, 0), limit=500):
        print("\nPreparing dataset for image classification...")
        if not project_paths:
            print("No image_path")
            return None
        # PrepareDataset.remove_empty_xml(xml_path)
        image_list, lbl_list = [], []
        for pr in project_paths:
            PrepareDataset.remove_empty_xml(f"{pr}/xml_labels")
            with os.scandir(f"{pr}/frames") as folder:
                for f in folder:
                    image_list.append((f"{pr}", f"{f.name}"))
            with os.scandir(f"{pr}/xml_labels") as folder:
                for f in folder:
                    lbl_list.append((f"{pr}", f"{f.name}"))
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
        for i, img in enumerate(image_list):
            if (labeled + not_labeled + 1) % int(length * 0.05) == 0:
                print(f"{round((labeled + not_labeled + 1) * 100 / length, 0)}% complete...")
            name = img[1].split(".")[0]
            pr_name = img[0].split("/")[-1]
            if (img[0], f"{name}.xml") in lbl_list:
                if resize != (0, 0):
                    image = Image.open(f"{img[0]}/frames/{img[1]}")
                    new_image = image.resize(resize)
                    new_image.save(f"{tmp_folder}/yes/{i}.png")
                else:
                    shutil.copy2(f"{img[0]}/frames/{img[1]}", f"{tmp_folder}/yes/{i}.png")
                labeled += 1
            else:
                if resize != (0, 0):
                    image = Image.open(f"{img[0]}/frames/{img[1]}")
                    new_image = image.resize(resize)
                    new_image.save(f"{tmp_folder}/no/{i}.png")
                else:
                    shutil.copy2(f"{img[0]}/frames/{img[1]}", f"{tmp_folder}/no/{i}.png")
                not_labeled += 1
            if labeled + not_labeled >= limit:
                break
        print(f"Prepare zip archive...")
        shutil.make_archive(f'{save_path}/{dataset_name}', 'zip', f"{tmp_folder}")
        shutil.rmtree(tmp_folder)
        print(f"Classificator dataset is ready! Images with boxes - {labeled}, images without boxes - {not_labeled}")
        print(f"Dataset path: '{f'{save_path}/{dataset_name}'}'. Preparation time={round(time.time() - st, 1)}s")

    @staticmethod
    def prepare_box_classification_dataset(project_paths: list, dataset_name: str,
                                           save_path: str, crop=0.2, limit=500):
        print("\nPreparing dataset for box classification...")
        if not project_paths:
            print("No image_path")
            return None
        # PrepareDataset.remove_empty_xml(xml_path)
        image_list, lbl_list = [], []
        for pr in project_paths:
            PrepareDataset.remove_empty_xml(f"{pr}/xml_labels")
            with os.scandir(f"{pr}/frames") as folder:
                for f in folder:
                    image_list.append((f"{pr}", f"{f.name}"))
            with os.scandir(f"{pr}/xml_labels") as folder:
                for f in folder:
                    lbl_list.append((f"{pr}", f"{f.name}"))
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

        count = 0
        random.shuffle(image_list)
        if limit > len(lbl_list):
            length = len(lbl_list)
        else:
            length = limit
        st = time.time()
        for i, img in enumerate(image_list):
            if (count + 1) % int(length * 0.05) == 0:
                print(f"{round((count + 1) * 100 / length, 0)}% complete...")
            name = img[1].split(".")[0]
            if (img[0], f"{name}.xml") in lbl_list:
                xml_info = PrepareDataset.read_xml(xml_path=f"{img[0]}/xml_labels/{name}.xml")
                bb = random.choice(xml_info['coords'])
                box_center = (bb[0] + int((bb[2]-bb[0])/2), bb[1] + int((bb[3]-bb[1])/2))
                box = True
            else:
                xml = random.choice(lbl_list)
                xml_info = PrepareDataset.read_xml(xml_path=f"{xml[0]}/xml_labels/{xml[1]}")
                bb = random.choice(xml_info['coords'])
                box_center = (bb[0] + int((bb[2] - bb[0]) / 2), bb[1] + int((bb[3] - bb[1]) / 2))
                box = False
            image = Image.open(f"{img[0]}/frames/{name}.png")
            w, h = image.size
            if box_center[0] < w / 2:
                left = int(box_center[0] - crop * w) if int(box_center[0] - crop * w) > 0 else 0
                right = int(box_center[0] + crop * w) if left > 0 else int(crop * w)
            else:
                right = int(box_center[0] + crop * w) if int(box_center[0] + crop * w) < w else w
                left = int(box_center[0] - crop * w) if right < w else int(w - crop * w)
            if box_center[1] < h / 2:
                top = int(box_center[1] - crop * h) if int(box_center[1] - crop * h) > 0 else 0
                bottom = int(box_center[1] + crop * h) if top > 0 else int(crop * h)
            else:
                bottom = int(box_center[1] + crop * h) if int(box_center[1] + crop * h) < h else h
                top = int(box_center[1] - crop * h) if bottom < h else int(h - crop * h)

            if box:
                yes_img = image.crop((left, top, right, bottom))
                yes_img.save(f"{tmp_folder}/yes/{i}.png")
            else:
                no_img = image.crop((left, top, right, bottom))
                no_img.save(f"{tmp_folder}/no/{i}.png")
            count += 1
            if count >= limit:
                break
        print(f"Prepare zip archive...")
        shutil.make_archive(f'{save_path}/{dataset_name}', 'zip', f"{tmp_folder}")
        shutil.rmtree(tmp_folder)
        print(f"Classificator dataset is ready! Images - {count}")
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
    pr_dir = [
        'datasets/Train_0_0s-300s',
        'datasets/Train_1_0s-300s',
        'datasets/Train_2_0s-300s',
        'datasets/Train_3_0s-300s',
        'datasets/Train_4_0s-300s',
    ]
    save_path = "datasets"

    # PrepareDataset.xml2terra_dataset(
    #     project_paths=pr_dir,
    #     dataset_name='complex_carpet_yolo_8000',
    #     save_path=save_path,
    #     shrink=True,
    #     limit=8000
    # )
    PrepareDataset.prepare_classificator_dataset(
        project_paths=pr_dir,
        dataset_name="complex_carpet_class_8000",
        resize=(416, 416),
        save_path=save_path,
        limit=8000
    )
    PrepareDataset.prepare_box_classification_dataset(
        project_paths=pr_dir,
        dataset_name="complex_box_class_8000",
        crop=0.1,
        save_path=save_path,
        limit=8000
    )
