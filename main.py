import cv2
import matplotlib.pyplot as plt
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not            # импортируем с файла util get_parking_spots_bboxes, empty_or_not 


def calc_diff(im1, im2):                                       # вычисляем среднее арефметическое np.abs(np.mean(im1) - np.mean(im2)) и выводим абсолютный результат               
    return np.abs(np.mean(im1) - np.mean(im2))                


mask = './data_and_model/mask_main.png'                               # относительный путь к маске
video_path = './data_and_model/parking_main_long.mp4'                 # путь к видео


mask = cv2.imread(mask, 0)                                              # считываем маску

cap = cv2.VideoCapture(video_path)                                      # считываем видео 

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S) # это метод OpenCV для нахождения связных компонент на бинарном изображении (маске).
                                                                             # 1. mask Исходное изображение с маской (обычно uint8 или CV_8U).
                                                  # 2. 4 — connectivity (связность) 4 → четырёхсвязность (пиксели соседями, если касаются по вертикали или горизонтали).
                                                                              # 8 → восьмисвязность (соседи и по диагонали тоже).
                                              # 3.cv2.CV_32S — тип данных для матрицы меток. Каждой компоненте присваивается уникальный целочисленный идентификатор (ID).
                                                                           #Здесь тип int32 (CV_32S) выбран, чтобы влезло достаточно разных меток.


spots = get_parking_spots_bboxes(connected_components)                   # Вызываем функцию из util и даем аргументы connected_components

spots_status = [None for j in spots]             # spots_status — список длины len(spots), полностью заполненный None. Позже в нём будут храниться статусы каждого спота.
                                                # diffs — аналогичный список, в который будут записываться значения «разницы» (diff) для каждого спота.
diffs = [None for j in spots]

previous_frame = None                           # будущий кадр пока None

frame_nmr = 0                                 # счетчик кадров
ret = True                                    # результат чтение кадра
step = 30                                     # FPS
while ret:
    ret, frame = cap.read()                   # создаем цикл while .read возвращет 2 значения успешность чтение + кадр. Считываем наше видео

    if frame_nmr % step == 0 and previous_frame is not None:    # условие этот код будет выполняться каждые 30 кадров начиная с нуля + проверка
                                                                 #previous_frame — это копия предыдущего кадра.
                                                                  #Если мы на первом кадре (previous_frame ещё пустой), то сравнивать не с чем → пропускаем.
                                                                      #Это защита от ошибки на первом шаге.          
        
        for spot_indx, spot in enumerate(spots):                        # spots — список прямоугольников (координат парковочных мест).
            x1, y1, w, h = spot                                          # x1, y1, w, h — координаты верхнего левого угла и размеры.

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]           # spot_crop — «вырезаем» текущее парковочное место из текущего кадра.

            diffs[spot_indx] = calc_diff(                      # diffs[spot_indx] — сохраняем разницу для этого места.
                                         # calc_diff(...) — сравниваем два изображения, чтобы понять, насколько они изменились (машина уехала, приехала и т.д.).
                spot_crop, 
                previous_frame[y1:y1 + h, x1:x1 + w, :]        #  # previous_frame[...] — вырезаем тот же участок из предыдущего кадра.
            )

        print([diffs[j] for j in np.argsort(diffs)][::-1])   # В консоль выводится список изменений по местам — от самых больших до самых маленьких.

    if frame_nmr % step == 0:
        if previous_frame is None:                            
            arr_ = range(len(spots))
        # Логика
#Если это первый обработанный кадр (previous_frame is None):
#У нас нет, с чем сравнить текущий кадр, поэтому алгоритм просто проверяет все споты.
#range(len(spots)) — создаёт диапазон от 0 до len(spots) - 1, то есть список индексов всех зон парковки/объектов.
#В результате arr_ будет вроде [0, 1, 2, 3, ...].
#Если previous_frame уже есть (все остальные кадры):
#Тогда переходим во вторую ветку кода и используем diffs для выбора только тех спотов, где есть заметные изменения.
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4] # np.argsort(diffs)  индексы спотов, отсортированные по возрастанию изменений 
                                                                                    # diffs[j] / np.amax(diffs) нормируем изменение j-го спота на максимум по всем
                                                                                 # > 0.4 берём только «заметно изменившиеся» (порог 40% от максимального)
            
        for spot_indx in arr_:                                            # перебираем индексы
            spot = spots[spot_indx]                                     # spots — список всех зон в кадре, каждая зона описана четырьмя числами:
            x1, y1, w, h = spot                                         # x1, y1 — координаты верхнего левого угла, w, h — ширина и высота зоны.
                                                                        # Мы достаём координаты конкретной зоны по её индексу.
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]                  # Вырезаем из текущего кадра (frame) именно ту область, которая соответствует зоне.
                                                                        # Здесь идёт обрезка по координатам и размерам: y1:y1+h и x1:x1+w.
                                                                        # Последний параметр : означает — берём все цветовые каналы (BGR).

            spot_status = empty_or_not(spot_crop)          # Передаём вырезанный фрагмент в функцию empty_or_not, которая определяет, пустая зона или занята
                                                                                                              # (например, парковочное место свободно или нет).
                                                           # Функция возвращает True (пусто) или False (занято).

            spots_status[spot_indx] = spot_status          # Обновляем глобальный список spots_status, записывая туда новое состояние конкретного спота.
                                                            # Теперь spots_status хранит актуальную картину занятости всех зон.

    if frame_nmr % step == 0:
        previous_frame = frame.copy()          # Используется копия (.copy()), чтобы последующие изменения frame (рисование рамок, текста) не попали в previous_frame.

    for spot_indx, spot in enumerate(spots):          # Идём по всем зонам (spots) и читаем:
        spot_status = spots_status[spot_indx]         #  spot_status — актуальный статус данного места из spots_status (True = свободно, False = занято).
        x1, y1, w, h = spots[spot_indx]               # Координаты прямоугольника спота: левый верхний угол (x1, y1) и размеры w, h. 

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)     # Отмечаем парковочные места зеленый значит свободно и красный нет
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)                            # Добавляем черный фон для текста на видео
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60), # сам текст
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)                # шрифт, размер шрифта,(белый цвет), толщина линий букв
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)                          # 'frame' — имя окна, которое появится на экране.
                                                                        # cv2.WINDOW_NORMAL — режим, в котором окно можно масштабировать (тянуть за края)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
