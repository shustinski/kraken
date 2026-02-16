from model.NeuralNetwork.dataset import NoCutDataset
from lib.images import *
from lib.data_interfaces import SampleGenerationSettings, CutSettings, TrainingParameters

import cProfile
import pstats
import io
import functools
import os
import datetime
from typing import Callable, Any


def profile(
    *,
    sort_by: str = "cumulative",   # "cumulative", "time", "calls", "filename", ...
    limit: int = 10,               # сколько строк вывести
    dump_to_file: bool = False,    # сохранять ли .prof‑файл
    filename_template: str = "{func_name}_{timestamp}.prof"
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Декоратор, который профилирует функцию с помощью cProfile и выводит
    отформатированную статистику через pstats.

    Параметры
    ----------
    sort_by : str
        Ключ сортировки, поддерживаемый pstats. Чаще всего используют
        "cumulative" (время в функции + в её подфункциях) или "time"
        (чистое время в функции).
    limit : int
        Сколько самых «тяжёлых» записей вывести. 0 – вывести всё.
    dump_to_file : bool
        Если True, статистика сохраняется в файл *.prof (можно загрузить позже
        в snakeviz, runsnakerun, gprof2dot и т.п.).
    filename_template : str
        Шаблон имени файла, в котором могут быть подставки:
        `{func_name}` – имя функции, `{timestamp}` – текущая дата‑время.

    Пример
    -------
    >>> @profile(sort_by="cumulative", limit=5, dump_to_file=False)
    ... def my_func():
    ...     ...

    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 1️⃣ Подготовка профайлера
            profiler = cProfile.Profile()
            try:
                # Запускаем профайлер **только** на время работы функции
                profiler.enable()
                result = func(*args, **kwargs)
                profiler.disable()
            finally:
                # 2️⃣ Сбор статистики в объект pstats
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s)
                ps.strip_dirs()                     # убираем лишние пути
                ps.sort_stats(sort_by)              # сортируем

                # 3️⃣ Вывод в консоль
                if limit:
                    ps.print_stats(limit)
                else:
                    ps.print_stats()

                # Печатаем готовый текст
                print(f"\n--- Профиль функции {func.__qualname__} ---")
                print(s.getvalue())

                # 4️⃣ При необходимости сохраняем в файл
                if dump_to_file:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = filename_template.format(
                        func_name=func.__name__, timestamp=timestamp
                    )
                    # Папка для файлов профайла (можно изменить)
                    out_dir = "profiles"
                    os.makedirs(out_dir, exist_ok=True)
                    full_path = os.path.join(out_dir, fname)
                    profiler.dump_stats(full_path)
                    print(f"Статистика сохранена в файл: {full_path}")

            # 5️⃣ Возвращаем оригинальный результат
            return result

        return wrapper

    return decorator


def test_image_parts_calculation():
    params = SampleGenerationSettings(step=10,
                                      x_size=16,
                                      y_size=16,
                                      vertical_rotation=True,
                                      horizontal_rotation=True,
                                      channels=1)
    path = 'D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/MSP430_M1. jpg/MSP430_M1_BS_00001.jpg'

    calc = SampleCalculator.from_path(path, params)
    print(f'Тест функции {test_image_parts_calculation.__name__}')
    print(path)
    print(params)
    print(len(calc))

def test_path_sample_calculation():
    params = CutSettings(step=16,
                         x_size=16,
                         y_size=16,
                         vertical_rotation=True,
                         horizontal_rotation=True,
                         color_mode='RGB',
                         model=Any
                         )
    path = 'D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/MSP430_M1. jpg'
    calc = SampleWorker(path, params)
    print(f'Тест функции {test_path_sample_calculation.__name__}')
    print(path)
    print(params)
    print(len(calc))

@profile()
def test_image_cut_on_fly():
    params = SampleGenerationSettings(step=10,
                                      x_size=16,
                                      y_size=16,
                                      vertical_rotation=True,
                                      horizontal_rotation=True,
                                      channels=1)
    image_path = 'D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/MSP430_M1. jpg/MSP430_M1_BS_00001.jpg'
    label_path = 'D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/binary cif/MSP430_M1_BS_00001.jpg'
    cutter = SampleFastCutter((image_path,label_path), params, shuffle=True)
    part = 0
    print()
    for img,lbl in cutter:
        part += 1
        # print(f'\r{part}', end='')
    print(part)

def test_image_preparation():
    image_path = 'D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/binary cif/MSP430_M1_BS_00736.jpg'
    params = SamplePrepareSettings(edge_cut=(10, 10), target_size=(1500, 1500))
    corrected_image = ImagePreparator(image_path, params).image

def test_image_preparation_and_cut():
    param_prep = SamplePrepareSettings(edge_cut=(10, 10), target_size=(1500, 1500))
    params_cut = SampleGenerationSettings(step=50,
                                      x_size=256,
                                      y_size=256,
                                      vertical_rotation=True,
                                      horizontal_rotation=True,
                                      channels=1)
    image_path = 'D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/MSP430_M1. jpg/MSP430_M1_BS_00001.jpg'
    label_path = 'D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/binary cif/MSP430_M1_BS_00001.jpg'
    corrected_image = ImagePreparator(image_path, param_prep).image
    corrected_label = ImagePreparator(label_path, param_prep).image
    cutter = SampleFastCutter.from_image((corrected_image, corrected_label), params_cut, shuffle=True)
    part = 0
    print()
    for img, lbl in cutter:
        part += 1
        print(f'\r{part}', end='')
    # print(part)

@profile()
def test_total_amount_of_samples():
    param_prep = SamplePrepareSettings(edge_cut=(10, 10), target_size=(1500, 1500))
    params_cut = SampleGenerationSettings(step=50,
                                          x_size=256,
                                          y_size=256,
                                          vertical_rotation=True,
                                          horizontal_rotation=True,
                                          channels=1)
    image_path = 'D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/MSP430_M1. jpg'
    images = filter_files(image_path, ('.jpg', '.bmp'))
    total_samples = 0
    for image in images:
        corrected_label = ImagePreparator(image, param_prep).size
        samples = len(SampleCalculator(corrected_label,params_cut))
        total_samples += samples
        print(total_samples)

def test_no_cut_dataset():
    image_path = Path('D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/jpg')
    label_path = Path('D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/bin')
    test_save_path = Path('D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/test_saves')
    param_prep = SamplePrepareSettings()
    params_cut = SampleGenerationSettings(step=1000,
                                          segment_size=(1000,1000),
                                          vertical_rotation=True,
                                          horizontal_rotation=True,
                                          channels=1)
    training_settings = TrainingParameters(image_path=image_path,
                                           label_path=label_path,
                                           prepare=param_prep,
                                           shuffle=True,
                                           generation=params_cut)
    dataset = NoCutDataset(training_settings)
    i = 0
    for image,label in dataset:
        save_binary(image,test_save_path/f'image_{i}.jpg')
        save_binary(label, test_save_path / f'label_{i}.jpg')
        i+=1






def run_image_tests():
    # test_image_parts_calculation()
    # test_path_sample_calculation()
    # test_image_cut_on_fly()
    # test_image_preparation()
    # test_image_preparation_and_cut()
    # test_total_amount_of_samples()
    test_no_cut_dataset()