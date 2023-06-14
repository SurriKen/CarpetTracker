from yolo8 import train

# train(epochs=100, weights='yolo8/yolov8n.pt', config='data_custom_CAM1.yaml', name='camera_1_mix_l+_8n_100ep')
# train(epochs=100, weights='yolo8/yolov8n.pt', config='data_custom_CAM2.yaml', name='camera_2_mix_l+_8n_100ep')
train(epochs=50, weights='runs/detect/camera_1_mix++_8n_150ep/weights/best.pt', config='data_custom_CAM1.yaml',
      name='camera_1_mix++_8n_200ep')
train(epochs=50, weights='runs/detect/camera_2_mix++_8n_150ep/weights/best.pt', config='data_custom_CAM2.yaml',
      name='camera_2_mix++_8n_200ep')

# train(epochs=50, weights='yolo8/yolov8n.pt', config='data_custom.yaml')
# train(epochs=50, weights='yolo8/yolov8s.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8s')
# train(epochs=50, weights='yolo8/yolov8m.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8m')
# train(epochs=50, weights='yolo8/yolov8l.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8l')
# train(epochs=50, weights='yolo8/yolov8x.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8x')

# train(epochs=50, weights='yolo8/yolov8s.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8s')
# train(epochs=50, weights='yolo8/yolov8m.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8m')
# train(epochs=50, weights='yolo8/yolov8l.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8l')
# train(epochs=50, weights='yolo8/yolov8x.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8x')

# train(epochs=50, weights='yolo8/yolov8s.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8s')
# train(epochs=50, weights='yolo8/yolov8m.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8m')
# train(epochs=50, weights='yolo8/yolov8l.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8l')
# train(epochs=50, weights='yolo8/yolov8x.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8x')
