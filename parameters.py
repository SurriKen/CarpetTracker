import os

ROOT_DIR = '/home/deny/Рабочий стол/CarpetTracker'

MIN_OBJ_SEQUENCE = 6
MIN_EMPTY_SEQUENCE = 12
SPEED_LIMIT_PERCENT = 0.2
DEAD_LIMIT_PERCENT = 0.005
GLOBAL_STEP = 0.25
CLASSES = sorted(['115x200', '115x400', '150x300', '60x90', '85x150'])
YOLO_WEIGTHS = {
    'model_1': os.path.join(ROOT_DIR, 'runs/camera_1/weights/best.pt'),
    'model_2': os.path.join(ROOT_DIR, 'runs/camera_2/weights/best.pt')
}
CLASSIFICATION_MODEL = os.path.join(ROOT_DIR, 'runs/classif_model/best.pt')


POLY_CAM1_IN = [[0.0573, 0.0], [0.2135, 0.6019], [0.3776, 0.3843], [0.2839, 0.0]]
POLY_CAM1_OUT = [[0.0, 0.0], [0.0, 0.4167], [0.1718, 0.6389], [0.2813, 0.7083], [0.3984, 0.5833], [0.4218, 0.4306],
                 [0.4141, 0.0]]
POLY_CAM2_IN = [[0.2187, 0.0], [0.2187, 0.4167], [0.4062, 0.5139], [0.4062, 0.0]]
POLY_CAM2_OUT = [[0.0938, 0.0], [0.1406, 0.5], [0.25, 0.6667],
                 [0.3906, 0.6944], [0.5156, 0.5277], [0.6718, 0.0]]