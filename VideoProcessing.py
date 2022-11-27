import os
import shutil

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from utils import save_dict, load_dict, get_colors


# from imageai.Detection import VideoObjectDetection


class VideoProcessing:

    def __init__(self):
        pass

    @staticmethod
    def cut_video(video_path: str, save_path: str = 'datasets', from_time=0, to_time=1000):
        try:
            os.mkdir(save_path)
        except:
            pass
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count / fps)
        saved_video_path = f"{save_path}/{video_path.split('/')[-1]}"
        if to_time and to_time < duration:
            clip = VideoFileClip(video_path).subclip(from_time, to_time)
            clip.write_videofile(saved_video_path)
        elif from_time:
            clip = VideoFileClip(video_path).subclip(from_time, duration)
            clip.write_videofile(saved_video_path)
        else:
            shutil.copy2(video_path, saved_video_path)
        print("Video was cut and save")
        return saved_video_path

    @staticmethod
    def video2frames(video_path: str, save_path: str = 'datasets', from_time=0, to_time=1000):
        video_name = video_path.split('/')[-1].split('.')[0]
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        # print(video_path, fps)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count / fps)
        if to_time:
            if to_time > int(duration):
                to_time = int(duration)
            to_path = f"{save_path}/{video_name}_{from_time}s-{to_time}s"
        elif from_time:
            to_path = f"{save_path}/{video_name}_{from_time}s-{duration}s"
        else:
            to_path = f"{save_path}/{video_name}_{from_time}s-{duration}s"
        try:
            os.mkdir(to_path)
        except:
            shutil.rmtree(to_path, ignore_errors=True)
            os.mkdir(to_path)
        os.mkdir(f"{to_path}/frames")
        os.mkdir(f"{to_path}/video")
        os.mkdir(f"{to_path}/xml_labels")
        saved_video_path = VideoProcessing.cut_video(
            video_path=video_path,
            save_path=f"{to_path}/video",
            from_time=from_time,
            to_time=to_time
        )
        video_capture = cv2.VideoCapture()
        video_capture.open(saved_video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Getting frames ({int(frames)} in total)...")
        size = ()
        for i in range(int(frames)):
            if (i + 1) % 200 == 0:
                print(f"{i + 1} frames are ready")
            ret, frame = video_capture.read()
            size = (frame.shape[1], frame.shape[0])
            cv2.imwrite(f"{to_path}/frames/%05d.png" % i, frame)
        video_data = {
            "fps": int(fps), "frames": int(frames), 'size': size
        }
        print(f"frames were got: fps - {int(fps)}, total frames - {int(frames)}, frame size - {size}")
        save_dict(video_data, to_path, 'data')

    @staticmethod
    def frames2video(frames_path: str, save_path: str, video_name: str,
                     params: str, box_path: str = None, resize=False, box_type='xml'):
        parameters = load_dict(params)
        image_list, box_list, lbl_list, color_list = [], [], [], []
        with os.scandir(frames_path) as folder:
            for f in folder:
                image_list.append(f.name)
        image_list = sorted(image_list)
        if box_path:
            PrepareDataset.remove_empty_xml(box_path)
            with os.scandir(box_path) as folder:
                for f in folder:
                    box_list.append(f.name)
            for box in box_list:
                box_info = PrepareDataset.read_xml(f"{box_path}/{f'{box}'}")
                if box_info["coords"][-1] not in lbl_list:
                    lbl_list.append(box_info["coords"][-1])
            lbl_list = sorted(lbl_list)
            color_list = get_colors(lbl_list)

        # tmp_folder = f"{save_path}/tmp"
        # try:
        #     os.mkdir(tmp_folder)
        # except:
        #     shutil.rmtree(tmp_folder, ignore_errors=True)
        #     os.mkdir(tmp_folder)

        st = time.time()
        count = 0
        print(f"Start processing...\n")
        out = cv2.VideoWriter(
            f'{save_path}/{video_name}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), parameters['fps'], parameters['size'])
        # print(f'{save_path}/{video_name}.mp4')
        for img in image_list:
            if (count + 1) % int(len(image_list) * 0.1) == 0:
                print(f"{round((count + 1) * 100 / len(image_list), 0)}% images were processed...")
            name = img.split(".")[0]
            if box_path and f"{name}.xml" in box_list:
                # print(img)
                if box_type == 'xml':
                    if resize:
                        box_info = PrepareDataset.read_xml(
                            xml_path=f"{box_path}/{f'{name}.xml'}",
                            shrink=True,
                            new_width=parameters['size'][0],
                            new_height=parameters['size'][1]
                        )
                    else:
                        box_info = PrepareDataset.read_xml(xml_path=f"{box_path}/{f'{name}.xml'}")
                    boxes, labels = [], []
                    # print(box_info)
                    for b in box_info["coords"]:
                        boxes.append(b[:-1])
                        labels.append(b[-1])
                    bbox = torch.tensor(boxes, dtype=torch.int)
                    image = read_image(f"{frames_path}/{img}")
                    image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)
                    image = torchvision.transforms.ToPILImage()(image_true)
                    # image = torchvision.transforms.ToPILImage()(image_true)
                    # image_true.save(f"{tmp_folder}/{img}")
                elif box_type == 'terra':
                    image = Image.open(f"{frames_path}/{img}")
                    # image = image.resize(parameters['size'])
                else:
                    image = Image.open(f"{frames_path}/{img}")
                    # image = image.resize(parameters['size'])
            else:
                if resize:
                    image = Image.open(f"{frames_path}/{img}")
                    image = image.resize(parameters['size'])
                    # new_image.save(f"{tmp_folder}/{img}")
                else:
                    image = Image.open(f"{frames_path}/{img}")
                    # shutil.copy2(f"{frames_path}/{img}", f"{tmp_folder}/{img}")
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            out.write(cv_img)
            count += 1
        out.release()
        print(f"\nProcessing is finished! Processing time = {round(time.time() - st, 1)}s\n")

        # out = cv2.VideoWriter(
        #     f'{save_path}/{video_name}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), parameters['fps'], parameters['size'])
        # count = 0
        # st = time.time()
        # print("Writing video is started...\n")
        # for filename in image_list:
        #     if (count + 1) % int(len(image_list) * 0.1) == 0:
        #         print(f"{round((count + 1) * 100 / len(image_list), 0)}% images were writed in video...")
        #     img = cv2.imread(f"{tmp_folder}/{filename}")
        #     out.write(img)
        #     count += 1
        # out.release()
        # print(f"Video is ready! Path to video: {f'{save_path}/{video_name}.mp4'}. "
        #       f"Writing time={round(time.time() - st, 1)}s")
        # shutil.rmtree(tmp_folder, ignore_errors=True)

    # @staticmethod
    # def video_detection(model_path, video_path, save_path, fps):
    #     https://wellsr.com/python/object-detection-from-videos-with-yolo/
    #     vid_obj_detect = VideoObjectDetection()
    #     vid_obj_detect.setModelTypeAsYOLOv3()
    #     vid_obj_detect.setModelPath(model_path) #.h5
    #     vid_obj_detect.loadModel()
    #     detected_vid_obj = vid_obj_detect.detectObjectsFromVideo(
    #         input_file_path=video_path,
    #         output_file_path=save_path,
    #         frames_per_second=fps,
    #         log_progress=True,
    #         return_detected_frame=True
    #     )
    #
    # @staticmethod
    # def make_video_with_boxes(frames_path: str, save_path: str, video_name: str,
    #                  params: str, box_path: str = None, resize=False, box_type='xml', obj_range=4, headline=False):
    #     PrepareDataset.remove_empty_xml(box_path)
    #     parameters = load_dict(params)
    #     st = time.time()
    #     # video_capture = cv2.VideoCapture()
    #     # video_capture.open(self.video_path)
    #     fps = parameters['fps']
    #     # frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #     height = parameters['size'][1]
    #     width = parameters['size'][0]
    #     size = (int(width), int(height))
    #     out = cv2.VideoWriter(
    #         f'{save_path}/{video_name}', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    #     obj_seq = []
    #     total_obj, count = 0, 0
    #     emp, obj = False, False
    #     # print(frame_count)
    #     for i in range(frame_count-1):
    #         ret, frame = video_capture.read()
    #         pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #         if CLASSIFICATION_MODEL_PATH:
    #             if self.classification_scaler == 'no_scaler':
    #                 class_array = np.expand_dims(np.array(pil_img.resize(self.target_class_size)), 0)
    #             else:
    #                 class_array = np.expand_dims(np.array(pil_img.resize(self.target_class_size)), 0) / 255
    #             result = self.class_model(class_array, training=False)
    #             result = int(np.argmax(result, -1)[0])
    #             if len(obj_seq) < obj_range:
    #                 obj_seq.append(result)
    #             else:
    #                 obj_seq.append(result)
    #                 obj_seq.pop(0)
    #         else:
    #             if len(obj_seq) < obj_range:
    #                 obj_seq.append(1)
    #             else:
    #                 obj_seq.append(1)
    #                 obj_seq.pop(0)
    #         total_obj, emp, obj = Predict.object_counter(obj_seq, emp, obj, obj_range, total_obj)
    #
    #         if headline:
    #             headline_str = f"Обнаружено объектов: {total_obj}"
    #         else:
    #             headline_str = ""
    #
    #         if obj_seq[-1] == 0:
    #             pil_img = Predict.add_headline_to_image(
    #                 image=pil_img,
    #                 headline=headline_str
    #             )
    #         else:
    #             img_array = pil_img.resize(self.target_yolo_size)
    #             if self.image_yolo_scaler == 'no_scaler':
    #                 img_array = np.expand_dims(np.array(img_array), 0)
    #             else:
    #                 img_array = np.expand_dims(np.array(img_array), 0) / 255
    #             predict = self.yolo_model(img_array, training=False)
    #             predict = self.get_yolo_y_pred(predict)
    #             bb = Predict.get_optimal_box_channel(predict)
    #             # print(img, bb, predict[bb])
    #             pil_img = Predict.plot_boxes(
    #                 pred_bb=predict[bb][0],
    #                 image=pil_img,
    #                 name_classes=self.classes_names,
    #                 colors=self.box_colors,
    #                 image_size=self.target_yolo_size,
    #                 headline=headline_str
    #             )
    #         # img = cv2.imread(f"{tmp_folder}/{filename}")
    #         img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    #         out.write(img)
    #         count += 1
    #         if count % int(frame_count * 0.01) == 0:
    #             print(
    #                 f"{int(count * 100 / frame_count)}% images were processed by tracker "
    #                 f"(frame: {i + 1}/{frame_count}, "
    #                 f"time: {int(round((time.time() - st) // 60), 0)}m {int(round((time.time() - st) % 60))}s)...")
    #
    #     out.release()
    #     print(f"\nPrediction time={round(time.time() - st, 1)}s")


if __name__ == '__main__':
    # for i in range(1):
    #     video_path_ = f'videos/Train_{i}.mp4'
    #     save_path_ = 'datasets'
    #     max_time = 10
    #     VideoProcessing.video2frames(
    #         video_path=video_path_,
    #         save_path=save_path_,
    #         from_time=0,
    #         to_time=max_time
    #     )
    # i=2
    # VideoProcessing.frames2video(
    #     frames_path=f'datasets/Train_{i}_0s-300s/frames',
    #     save_path=f'datasets/Train_{i}_0s-300s',
    #     video_name=f'Train_{i}_with_boxes_2',
    #     params=f'datasets/Train_{i}_0s-300s/data.dict',
    #     box_path=f'datasets/Train_{i}_0s-300s/xml_labels',
    #     resize=False
    # )
    VideoProcessing.cut_video(
        video_path=f'videos/Train_0.mp4',
        save_path=f'datasets/Train_0_0s-15s',
        from_time=0,
        to_time=15
    )
