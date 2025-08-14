import pickle

from skimage.transform import resize
import numpy as np
import cv2


EMPTY = True                                                                         # ставим булевые значения для наших парковачных мест
NOT_EMPTY = False

MODEL = pickle.load(open("./data_and_model/model_parking_detection.p", "rb"))       # берем нашу модель 


def empty_or_not(spot_bgr):                                              # функция для определение пустое парковачное место ил нет

    flat_data = []                                                       # пустой список енужен будет позже

    img_resized = resize(spot_bgr, (15, 15, 3))                          # делаем ресайз изображения 
    flat_data.append(img_resized.flatten())                    # Метод .flatten() из NumPy превращает многомерный массив (2D или 3D) в одномерный вектор признаков
    flat_data = np.array(flat_data)                                      # переводим в массив нумпай

    y_output = MODEL.predict(flat_data)                                  # делаем предсказания 

    if y_output == 0:                                                    # если предсказания равно 0 то есть на парковке нет машин, значит EMPTYU 
        return EMPTY
    else:
        return NOT_EMPTY                                                 # NOT_EMPTY в других ситуациях 


def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components  
                            #  Эта функция берёт результаты cv2.connectedComponentsWithStats и переводит их в список координат парковочных мест.
                                   # totalLabels — общее количество найденных областей (включая фон, который всегда индекс 0).
                                   # label_ids — матрица с теми же размерами, что и входное изображение, где каждый пиксель имеет ID объекта, которому он принадлежит.
                                   # values — таблица статистики для каждого объекта:
                                   # centroid — координаты центров объектов (не используются в этом коде).

    slots = []
    coef = 1
    for i in range(1, totalLabels):        # Индекс 0 всегда — это фон, поэтому цикл начинается с 1.

        # Now extract the coordinate points                # Здесь coef (равен 1) можно использовать для масштабирования, если изображение было уменьшено или увеличено.
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)       # Преобразование в int — чтобы координаты были целыми числами.
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
            #values[i, cv2.CC_STAT_LEFT] — x-координата левого края.
            #values[i, cv2.CC_STAT_TOP] — y-координата верхнего края.
            #values[i, cv2.CC_STAT_WIDTH] — ширина объекта.
            #values[i, cv2.CC_STAT_HEIGHT] — высота объекта.
        slots.append([x1, y1, w, h])     # добавляем координаты

    return slots     # возвращаем координаты 

