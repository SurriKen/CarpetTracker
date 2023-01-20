import argparse
import os

os.system('!pip install pillow')
from PIL import Image
import numpy as np


# Основная функция
def prepare_image(path):
    print(path)
    img = Image.open(path)  # Загружаем картинку по переданному пути
    print(f'Размер исходных данных: {np.array(img).shape}')  # Выводим информацию о размере исходного изображения

    img = img.resize((28, 28)).convert('L')
    img = np.array(img)  # Преобразуем изображение в numpy-массив
    print(img.dtype)
    img = 255 - np.where(img >= 128, 255, 0)
    # img = img.reshape(1, -1)  # Вытягиваем в вектор
    # img = img.astype('float32') / 255.  # Нормализуем данные
    print(f'Размер данных после преобразования: {img.shape}')  # Выводим информацию о новом размере данных

    # img = np.expand_dims(img, axis=-1)
    # print(img.shape)
    # img = np.concatenate([img, img, img], axis=-1) * 255
    img = img.astype('uint8')
    print(img.shape)
    img2 = Image.fromarray(img)
    img2.save('xxx.png')
    # with open('prepared_image.npy', 'wb') as f:  # Создаем новый файл для записи данных
    #     np.save(f, img)  # Записываем в файл обработанный numpy-массив

    print('Изображение обработано!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Greetings')  # Создаем парсер для аргументов, переданных через командную строку
    parser.add_argument('path', type=str, help='Image path')  # Задаем один обязательный аргумент - имя
    args = parser.parse_args()  # Парсим аргументы
    path = args.path  # Извлекаем имя
    # path = 'test.jpg'
    prepare_image(path)  # Передаем имя в функцию для приветствия
