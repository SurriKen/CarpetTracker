# from dataset_processing import DatasetProcessing
#
# generator_dict = DatasetProcessing.create_video_class_dataset_generator(folder_path='datasets/class_videos', split=0.9)
# print(generator_dict.keys())
# print(generator_dict.get('stat'))
# print(generator_dict.get('x_train')[:10])
# print(generator_dict.get('y_train')[:10])
#
# x_batch, y_batch = DatasetProcessing.generate_video_class_batch(generator_dict=generator_dict, iteration=0, mode='train')
# print(x_batch.shape, y_batch.shape, y_batch)
from random import shuffle
x = 125
c = []
for i in range(10):
    c.append((int(x * (i + 1) / 10), '10 %'))

print(c)