import os
import shutil

from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import *

from lib import System

from UI.my_ui_elements import *

from UI.UI_design import *

import tkinter as tk
from tkinter import filedialog
from lib import backend

from lib.config import Config

import enum

from PIL import Image


class SampleSettings(QWidget):
    log_data = pyqtSignal(object)
    samples_ready = pyqtSignal(object)
    gui_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.label_folder = None
        self.image_folder = None
        self.convert_thread = None
        self.cut_thread = None

        self.model_type = None

        self.jpeg_files = 0

        self.sample_prep_automat = SamplePrepSteps

        self.segment_size = (256, 256)

        # self.setTitle("Настройка параметров выборки:")

        self.grid = QGridLayout()
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 100)
        self.setLayout(self.grid)

        y_grid_loc = 0

        jpg_path_text = QLabel("Путь до JPG")
        self.grid.addWidget(jpg_path_text, y_grid_loc, 0)

        self.jpg_path_label = ClickableLabel()
        # self.sample_path.setStyleSheet(GuiStyle.path_label_style)
        self.jpg_path_label.setToolTip("Укажите путь к кадрам для выборки")
        self.jpg_path_label.clicked.connect(lambda: self.select_source())
        self.grid.addWidget(self.jpg_path_label, y_grid_loc, 1)
        y_grid_loc += 1

        cif_path_text = QLabel("Путь до CIF")
        self.grid.addWidget(cif_path_text, y_grid_loc, 0)

        self.cif_path_label = ClickableLabel()
        # self.label_path.setStyleSheet(GuiStyle.path_label_style)
        self.cif_path_label.setToolTip("Укажите путь к кадрам для выборки")
        self.cif_path_label.clicked.connect(lambda: self.select_source())
        self.grid.addWidget(self.cif_path_label, y_grid_loc, 1)
        y_grid_loc += 1

        self.sample_parameters_groupbox = QGroupBox("Настройки генерации выборки:")
        self.grid.addWidget(self.sample_parameters_groupbox, y_grid_loc, 0, 1, 2)
        self.parameters_layout = QGridLayout()
        self.sample_parameters_groupbox.setLayout(self.parameters_layout)

        self.enable_horizontal_rotation = QCheckBox()
        self.enable_horizontal_rotation.setText("Поворот кадра на 90 градусов")
        self.enable_horizontal_rotation.clicked.connect(lambda: self.calculate_samples_number())
        self.enable_horizontal_rotation.setToolTip("Разрешить поворачивать кадры выборки на угол =90,+90 градусов")
        self.parameters_layout.addWidget(self.enable_horizontal_rotation, 0, 0)

        self.enable_vertical_rotation = QCheckBox()
        self.enable_vertical_rotation.setText("Поворот кадра на 180 градусов")
        self.enable_vertical_rotation.setToolTip("Разрешить поворачивать кадры выборки на угол 180градусов")
        self.enable_vertical_rotation.clicked.connect(lambda: self.calculate_samples_number())
        self.parameters_layout.addWidget(self.enable_vertical_rotation, 0, 1)

        self.cut_step_label = QLabel("Смещение")
        self.cut_step_label.setToolTip("Шаг через который будет происходить нарезка кадра")
        self.parameters_layout.addWidget(self.cut_step_label, 1, 0)

        self.cut_step_slider = QSlider(Qt.Orientation.Horizontal)
        self.cut_step_slider.setMinimum(1)
        self.cut_step_slider.setMaximum(25)
        self.cut_step_slider.setValue(25)
        self.cut_step_slider.setTickInterval(1)
        self.cut_step_slider.setSingleStep(1)
        self.cut_step_slider.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.cut_step_slider.valueChanged.connect(lambda: self.slider_shifted())
        self.cut_step_slider.setToolTip("Шаг через который будет происходить нарезка кадра")
        self.parameters_layout.addWidget(self.cut_step_slider, 1, 1, 1, 2)

        self.current_cut_step_label = QLabel("Текущее смещение: 256")
        self.parameters_layout.addWidget(self.current_cut_step_label, 2, 1)

        self.samples_number = QLabel("Кадров в выборке: 0")
        self.parameters_layout.addWidget(self.samples_number, 2, 2)

        self.restore()

    def slider_shifted(self):
        self.cut_step = 256 if self.cut_step_slider.value() == 25 else 10 * self.cut_step_slider.value()
        self.current_cut_step_label.setText(f"Текущее смещение: {self.cut_step}")
        self.calculate_samples_number()

    def select_source(self):
        sender = self.sender()
        sender.setEnabled(False)
        root = tk.Tk()
        root.withdraw()
        directory = filedialog.askdirectory()

        if directory is None or directory == "":
            sender.setEnabled(True)
            return

        self.jpeg_files = len(os.listdir(directory))

        sender.setText(directory)
        sender.setEnabled(True)
        if self.can_be_started():
            self.gui_ready.emit(True)

        self.calculate_samples_number()

    def calculate_samples_number(self):
        if self.jpeg_files == 0:
            self.samples_number.setText("Кадров в выборке: 0")
            self.samples_number.setToolTip("В выборке недостаточно кадров")
            # self.samples_number.setStyleSheet(GuiStyle.insufficient_samples_style)
            # self.samples_number.setFont(GuiStyle.font)
            return

        files_in_folder = os.listdir(self.jpg_path_label.text())
        first_file = os.path.join(self.jpg_path_label.text(), files_in_folder[0])

        with Image.open(first_file) as frame:
            im_height, im_width = frame.size

            width_steps = int(im_width / self.cut_step) + 1
            height_steps = int(im_height / self.cut_step) + 1

            frames_in_frame = width_steps * height_steps

            vertical_frames = 0
            horizontal_frames = 0
            if self.enable_vertical_rotation.isChecked():
                vertical_frames = frames_in_frame

            if self.enable_horizontal_rotation.isChecked():
                horizontal_frames = 2 * frames_in_frame

            frames_in_frame += horizontal_frames + vertical_frames

        total_frames = frames_in_frame * len(files_in_folder)
        self.samples_number.setText(f"Кадров в выборке: {format(total_frames, ',')}")

        if self.model_type is None:
            return

        if self.model_type == 'small':
            if total_frames < 20000:
                self.samples_number.setStyleSheet(GuiStyle.insufficient_samples_style)
                self.samples_number.setToolTip("В выборке недостаточно кадров. Установите количество кадров в "
                                               "диапазоне 20,000 - 80,0000")
            elif total_frames < 80000:
                self.samples_number.setStyleSheet(GuiStyle.sufficient_samples_style)
                self.samples_number.setToolTip("Оптимальное количество кадров")
            else:
                self.samples_number.setStyleSheet(GuiStyle.insufficient_samples_style)
                self.samples_number.setToolTip("Слишком много кадров. Установите количество кадров в "
                                               "диапазоне 20,000 - 80,0000")
        elif self.model_type == 'medium':
            if total_frames < 200000:
                self.samples_number.setStyleSheet(GuiStyle.insufficient_samples_style)
                self.samples_number.setToolTip("В выборке недостаточно кадров. Установите количество кадров в "
                                               "диапазоне 200,000 - 500,0000")
            elif total_frames < 500000:
                self.samples_number.setStyleSheet(GuiStyle.sufficient_samples_style)
                self.samples_number.setToolTip("Оптимальное количество кадров")
            else:
                self.samples_number.setStyleSheet(GuiStyle.insufficient_samples_style)
                self.samples_number.setToolTip("Слишком много кадров. Установите количество кадров в "
                                               "диапазоне 200,000 - 500,0000")
        else:
            self.samples_number.setStyleSheet(GuiStyle.unknown_samples_style)
            self.samples_number.setToolTip("Вы выбрали настраиваемую модель. "
                                           "Количество кадров для выборки выбирайте эмпирически")

    def do_samples(self):
        self.sample_prep_automat = SamplePrepSteps.convert_cif
        self.preparation_automat()

    def get_cif_folder(self):
        return self.cif_path_label.text()

    def set_model_type(self, model):
        self.model_type = model
        self.calculate_samples_number()

    def preparation_automat(self):
        match self.sample_prep_automat:
            case SamplePrepSteps.wait:
                return
            case SamplePrepSteps.convert_cif:
                self.convert_cif_to_jpg()
                self.sample_prep_automat = SamplePrepSteps.cut_cif
                return
            case SamplePrepSteps.cut_cif:
                self.emit_log_data("Преобразование кадров завершено")
                input_path = os.path.join(self.get_temporary_folder(), "binary_cif")
                self.label_folder = os.path.join(self.get_temporary_folder(), "label_dir")
                self.sample_prep_automat = SamplePrepSteps.cut_jpg
                self.cut_image(input_path, self.label_folder)
            case SamplePrepSteps.cut_jpg:
                self.emit_log_data("Нарезка векторов завершена")
                self.image_folder = os.path.join(self.get_temporary_folder(), "input_dir")
                self.sample_prep_automat = SamplePrepSteps.finish
                self.cut_image(self.jpg_path_label.text(), self.image_folder)
            case SamplePrepSteps.finish:
                self.emit_log_data("Нарезка изображений завершена")
                self.samples_ready.emit(1)
                return

    def convert_cif_to_jpg(self):
        savefolder = os.path.join(self.get_temporary_folder(), "binary_cif")
        self.convert_thread = ConvertCifThread(self.get_cif_folder(), savefolder)
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)
        self.convert_thread.current_frame.connect(self.emit_log_data)
        self.convert_thread.finish_signal.connect(self.thread_finished)
        self.convert_thread.start()

    def cut_image(self, input_path, save_path):
        if os.path.exists(save_path) and os.listdir(save_path):
            question = QMessageBox.question(self, 'Предупреждение',
                                            f'При попытке нарезки кадров из папки {input_path} было обгпружено, '
                                            f'что в {save_path} уже есть файлы. Нажмите Yes чтобы использовать их и No, '
                                            f'чтобы произвести нарезку заново',
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                            QMessageBox.StandardButton.No)
            if question == QMessageBox.StandardButton.Yes:
                self.thread_finished()
                return
        try:
            if os.path.exists(save_path):
                self.log_data.emit(f'Идет удаление папки{save_path}')
                shutil.rmtree(save_path)
        except PermissionError:
            WarningMsgBox("Не удалось удалить папку, перезапустите программу с правами администратора")
            return

        os.mkdir(save_path)

        self.cut_thread = CutPictureThread(path=input_path,
                                           savepath=save_path,
                                           segment_size=self.segment_size,
                                           horizontal_rotation=self.enable_horizontal_rotation.isChecked(),
                                           vertical_rotation=self.enable_vertical_rotation.isChecked(),
                                           cut_step=self.cut_step)
        self.cut_thread.current_frame.connect(self.emit_log_data)
        self.cut_thread.finish_signal.connect(self.thread_finished)
        self.cut_thread.start()

    def save_config(self):
        Config.set_data('cif_sample_jpg_path', self.jpg_path_label.text())
        Config.set_data('cif_sample_cif_path', self.cif_path_label.text())
        Config.set_data('rotation_90', self.enable_horizontal_rotation.isChecked())
        Config.set_data('rotation_180', self.enable_vertical_rotation.isChecked())
        Config.set_data('shift', self.cut_step_slider.value())

    def restore(self):
        self.jpg_path_label.setText(Config.get_data('cif_sample_jpg_path'))
        self.cif_path_label.setText(Config.get_data('cif_sample_cif_path'))
        self.enable_horizontal_rotation.setChecked(Config.get_data('rotation_90'))
        self.enable_vertical_rotation.setChecked(Config.get_data('rotation_180'))
        self.cut_step_slider.setValue(Config.get_data('shift'))
        self.slider_shifted()
        path = self.jpg_path_label.text()
        if  path!= '' and os.path.isdir(path):
            self.jpeg_files = len(os.listdir(self.jpg_path_label.text()))
            self.calculate_samples_number()

    def thread_finished(self):
        self.preparation_automat()

    def get_temporary_folder(self):
        savefolder = os.path.split(self.cif_path_label.text())[0]
        return savefolder

    def get_image_folder(self):
        return self.image_folder

    def get_label_folder(self):
        return self.label_folder

    def get_model_name_part(self):
        name = ''
        if self.enable_horizontal_rotation.isChecked():
            name += 'rot90'
        if self.enable_vertical_rotation.isChecked():
            name += 'rot180'
        name += f'_shift{self.cut_step}'
        return name

    def emit_log_data(self, data):
        self.log_data.emit(data)

    def can_be_started(self):
        if self.cif_path_label.text() and self.jpg_path_label.text():
            return 1
        return 0


class ConvertCifThread(QThread):
    current_frame = pyqtSignal(object)
    finish_signal = pyqtSignal(object)

    def __init__(self, path: str, savepath: str):
        super().__init__()
        self.path = path
        self.savepath = savepath

    def run(self):
        files_in_folder = os.listdir(self.path)
        self.current_frame.emit("Начинаю преобразование cif в бинарные изображения")
        for file in files_in_folder:
            filename, extension = os.path.splitext(file)
            if extension.lower() != '.cif':
                continue
            self.current_frame.emit(f"Преобразую в jpg файл {filename}")
            cif_path = os.path.join(self.path, file)
            imaged_cif = backend.cif_to_jpg(cif_path)
            save_name = os.path.splitext(os.path.basename(cif_path))[0] + '.jpg'
            save_path = os.path.join(self.savepath, save_name)
            imaged_cif.save(save_path)

        self.finish_signal.emit(1)


class CutPictureThread(QThread):
    current_frame = pyqtSignal(object)
    finish_signal = pyqtSignal(object)

    def __init__(self, path: str, savepath: str, segment_size: tuple[int, int],
                 horizontal_rotation: bool, vertical_rotation: bool, cut_step: int):
        super().__init__()
        self.path = path
        self.savepath = savepath
        self.segment_size = segment_size
        self.horizontal_rotation = horizontal_rotation
        self.vertical_rotation = vertical_rotation
        self.cut_step = cut_step

    def run(self):
        files_in_folder = os.listdir(self.path)
        self.current_frame.emit("Начинаю производить нарезку кадров")
        for file in files_in_folder:
            filename, extension = os.path.splitext(file)
            if extension.lower() != '.jpg':
                continue
            self.current_frame.emit(f"Обрабатывается файл {filename}")
            backend.frame_cut(os.path.join(self.path, file), self.savepath, self.segment_size, self.horizontal_rotation,
                              self.vertical_rotation, self.cut_step)
        self.finish_signal.emit(1)


@enum.unique
class SamplePrepSteps(enum.Enum):
    wait = 0
    convert_cif = 1
    cut_cif = 2
    cut_jpg = 3
    finish = 4


class ModelSettingsUI(QGroupBox):
    current_model = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(self)

        y_location = 0

        self.model_type_groupbox = QGroupBox()
        self.model_type_groupbox.setTitle('Тип модели нейронной сети:')
        self.model_type_grid = QHBoxLayout(self.model_type_groupbox)

        self.model_small = QRadioButton('Маленькая')
        self.model_type_grid.addWidget(self.model_small)
        self.model_small.clicked.connect(lambda: self.model_type_changed())
        self.model_small.setChecked(True)

        self.model_medium = QRadioButton('Средняя')
        self.model_medium.clicked.connect(lambda: self.model_type_changed())
        self.model_type_grid.addWidget(self.model_medium)

        self.model_customizable = QRadioButton('Настраиваемая')
        self.model_customizable.clicked.connect(lambda: self.model_type_changed())
        self.model_type_grid.addWidget(self.model_customizable)

        self.grid.addWidget(self.model_type_groupbox, y_location, 0)

        self.image_type_groupbox = QGroupBox()
        self.image_type_groupbox.setTitle('Формат изображений:')
        self.image_type_grid = QHBoxLayout(self.image_type_groupbox)

        self.grayscale_image_type = QRadioButton('Оттенки серого')
        self.image_type_grid.addWidget(self.grayscale_image_type)
        self.grayscale_image_type.setChecked(True)

        self.rgb_image_type = QRadioButton('RGB')
        self.image_type_grid.addWidget(self.rgb_image_type)

        self.grid.addWidget(self.image_type_groupbox, y_location, 1)

        y_location += 1

        self.model_parameters = QGroupBox()
        self.model_parameters_layout = QFormLayout(self.model_parameters)

        self.model_layers_lineedit = QLineEdit('3')
        self.model_layers_lineedit.setInputMask("00")
        self.model_layers_lineedit.setToolTip("Число слоев в нейронной сети. Маленькая содержит 3 слоя, средняя - 5")
        self.model_parameters_layout.addRow('Число слоев:', self.model_layers_lineedit)

        self.model_start_filter_lineedit = QLineEdit('32')
        self.model_start_filter_lineedit.setInputMask("000")
        self.model_start_filter_lineedit.setToolTip("Число фильтров для первого слоя нейронной сети. Для маленькой "
                                                    "сети этот параметр равен 32, а для средней - 64")
        self.model_parameters_layout.addRow('Фильтров первого слоя:', self.model_start_filter_lineedit)

        self.model_filter_shift_lineedit = QLineEdit('32')
        self.model_filter_shift_lineedit.setInputMask("000")
        self.model_filter_shift_lineedit.setToolTip("На сколько будет увеличиваться число фильтров на каждом "
                                                    "следующем слов. Для маленькой и средней нейронной сети этот "
                                                    "параметр равен 32")
        self.model_parameters_layout.addRow('Изменение фильтра:', self.model_filter_shift_lineedit)

        self.grid.addWidget(self.model_parameters, y_location, 0, 1, 2)
        self.model_parameters.setEnabled(False)

        y_location += 1

        validation_groupbox = QGroupBox("Валидация:")
        validation_layout = QHBoxLayout(validation_groupbox)

        self.validation_checkbox = QCheckBox('Включить')
        validation_layout.addWidget(self.validation_checkbox)

        valid_coef_label = QLabel('Процент выборки на валидацию:')
        validation_layout.addWidget(valid_coef_label)

        self.validation_coef_lineedit = QLineEdit()
        self.validation_coef_lineedit.setInputMask("000")
        validation_layout.addWidget(self.validation_coef_lineedit)

        self.grid.addWidget(validation_groupbox, y_location, 0, 1, 2)

        y_location += 1

        gpus = System.check_gpu_availability()

        if gpus > 1:
            gpu_usage_params_groupbox = QGroupBox()
            gpu_usage_params_groupbox.setEnabled(False)

            training_params_layout = QVBoxLayout(gpu_usage_params_groupbox)

            self.use_multiple_gpu_while_training = QCheckBox('Использовать несколько видеокарт во время обучения')
            self.use_multiple_gpu_while_training.setToolTip(
                'Выберите эту опцию для обучения бошьших моделей и для ускорения обучения')
            training_params_layout.addWidget(self.use_multiple_gpu_while_training)

            multiple_gpu_while_predict_groupbox = QGroupBox()
            multiple_gpu_while_predict_groupbox.setTitle('Использовать нескольких видеокарт во время распознавания:')
            training_params_layout.addWidget(multiple_gpu_while_predict_groupbox)

            multiple_gpu_while_predict_layout = QHBoxLayout(multiple_gpu_while_predict_groupbox)

            self.not_use_multiple_gpu_predict = QRadioButton('Не использовать')
            multiple_gpu_while_predict_layout.addWidget(self.not_use_multiple_gpu_predict)

            self.load_same_model_to_gpu = QRadioButton('Загрузить модели параллельно')
            self.load_same_model_to_gpu.setToolTip('Используйте эту опцию с небольшими '
                                                   'моделями чтобы ускорить расрознавание')
            multiple_gpu_while_predict_layout.addWidget(self.load_same_model_to_gpu)

            self.load_one_model_to_gpus = QRadioButton('Загрузить одну модель на несколько видеокарт')
            self.load_one_model_to_gpus.setToolTip(
                'Используйте эту опцию с большими модеями, которые не помещаются на одну видеокарту')
            multiple_gpu_while_predict_layout.addWidget(self.load_one_model_to_gpus)

            self.grid.addWidget(gpu_usage_params_groupbox, y_location, 0, 1, 2)

        y_location += 1

        self.restore()

    def model_type_changed(self):
        if self.model_small.isChecked():
            self.model_layers_lineedit.setText("3")
            self.model_start_filter_lineedit.setText("32")
            self.model_filter_shift_lineedit.setText("32")
            self.model_parameters.setEnabled(False)
            self.current_model.emit('small')
        elif self.model_medium.isChecked():
            self.model_layers_lineedit.setText("5")
            self.model_start_filter_lineedit.setText("64")
            self.model_filter_shift_lineedit.setText("32")
            self.model_parameters.setEnabled(False)
            self.current_model.emit('medium')
        else:
            self.model_parameters.setEnabled(True)
            self.current_model.emit('customizable')

    def get_state(self):
        if not self.check_correctness():
            return 0

        if self.model_small.isChecked():
            model_type = 'small'
        elif self.model_medium.isChecked():
            model_type = 'medium'
        else:
            model_type = 'customisable'

        state = {
            'type': model_type,
            'layers': int(self.model_layers_lineedit.text()),
            'start_filter': int(self.model_start_filter_lineedit.text()),
            'step_filter': int(self.model_filter_shift_lineedit.text()),
            'use_validation': bool(self.validation_checkbox.isChecked()),
            'validation_coeff': int(self.validation_coef_lineedit.text()) if self.validation_coef_lineedit.text() else 0,
            'channels': 1 if self.grayscale_image_type.isChecked() else 3,
            'sample_width': int(self.sample_width_lineedit.text()),
            'sample_height': int(self.sample_height_lineedit.text())
        }

        return state

    @property
    def channels(self):
        return 1 if self.grayscale_image_type.isChecked() else 3

    @property
    def image_x_size(self):
        return int(self.sample_width_lineedit.text()) if self.sample_width_lineedit.text() else 0

    @property
    def image_y_size(self):
        return int(self.sample_height_lineedit.text()) if self.sample_height_lineedit.text() else 0

    def check_correctness(self):
        if not (self.model_layers_lineedit.text() or
                self.model_start_filter_lineedit.text() or
                self.model_filter_shift_lineedit.text()):
            WarningMsgBox('Заполните все поля')
            return 0

        if int(self.model_layers_lineedit.text()) == 0 or int(self.model_start_filter_lineedit.text()) == 0:
            WarningMsgBox('Слоев и фильтров не может быть 0')
            return 0

        if not self.sample_width_lineedit.text() or not self.sample_height_lineedit.text():
            WarningMsgBox('Поля с размерами не могут быть пустыми')
            return 0

        if not int(self.sample_width_lineedit.text()):
            WarningMsgBox('Размер не может быть нулевой')
            return 0

        if not int(self.sample_height_lineedit.text()):
            WarningMsgBox('Размер не может быть нулевой')
            return 0

        return 1

    def save_config(self):
        model_type = "configurable"
        if self.model_small.isChecked():
            model_type = 'small'
        elif self.model_medium.isChecked():
            model_type = 'medium'
        Config.set_data("model_type", model_type)

    def restore(self):
        match Config.get_data('model_type'):
            case 'small':
                self.model_small.setChecked(True)
            case 'medium':
                self.model_medium.setChecked(True)
            case 'configurable':
                self.model_customizable.setChecked(True)

        self.model_type_changed()


class LearningSettingsWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.setStyleSheet(GuiStyle.style_sheet)
        self.setWindowTitle("Параметры обучения")
        self.setWindowIcon(QtGui.QIcon('_internal/icon.png'))

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        y_location = 0
        self.samples_preparation_groupbox = QGroupBox()
        self.layout.addWidget(self.samples_preparation_groupbox, y_location, 0, 1, 2)
        self.samples_preparation_groupbox.setTitle("Метод создания выборки:")


        groupbox_layout = QHBoxLayout()
        self.samples_preparation_groupbox.setLayout(groupbox_layout)
        self.image_cut_preparation = QRadioButton("Нарезка в файл")
        groupbox_layout.addWidget(self.image_cut_preparation)
        self.no_cut_preparation = QRadioButton("Без нарезки")
        groupbox_layout.addWidget(self.no_cut_preparation)

        y_location += 1

        self.image_size_groupbox = QGroupBox()
        image_size_grid = QHBoxLayout(self.image_size_groupbox)
        image_size_grid.setStretch(0, 3)
        image_size_grid.setStretch(1, 2)
        image_size_grid.setStretch(2, 1)
        image_size_grid.setStretch(3, 2)

        image_size_grid.addWidget(QLabel("Размер изображения:"))

        self.sample_width_lineedit = QLineEdit('256')
        self.sample_width_lineedit.setToolTip('Размер изображения по горизонтали')
        self.sample_width_lineedit.setInputMask("0000")
        image_size_grid.addWidget(self.sample_width_lineedit)

        image_size_grid.addWidget(QLabel('X'))

        self.sample_height_lineedit = QLineEdit('256')
        self.sample_height_lineedit.setToolTip('Размер изображения по вертикали')
        self.sample_height_lineedit.setInputMask("0000")
        image_size_grid.addWidget(self.sample_height_lineedit)

        self.layout.addWidget(self.image_size_groupbox, y_location, 0, 1, 2)

        y_location += 1

        self.ok_button = QPushButton("Ок")
        self.layout.addWidget(self.ok_button, y_location, 0, 1,2)
        self.ok_button.clicked.connect(lambda : self.ok_clicked())

        self.restore()

    def ok_clicked(self):
        self.save()
        self.close()

    def restore(self):
        match Config.get_data("sample_preparation_type"):
            case 'cut':
                self.image_cut_preparation.setChecked(True)
            case 'no_cut':
                self.no_cut_preparation.setChecked(True)

        self.sample_width_lineedit.setText(str(Config.get_data('sample_wight')))
        self.sample_height_lineedit.setText(str(Config.get_data('sample_height')))

    def save(self):
        if self.image_cut_preparation.isChecked():
            Config.set_data('sample_preparation_type', 'cut')
        if self.no_cut_preparation.isChecked():
            Config.set_data('sample_preparation_type', 'no_cut')

        width = int(self.sample_width_lineedit.text())
        height = int(self.sample_height_lineedit.text())
        Config.set_data('sample_wight', width)
        Config.set_data('sample_height', height)

        Config.save()

        print(self.__class__.__name__, " saved")

    def closeEvent(self, event):
        event.accept()





if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    # window = CutSettings()
    # window = ModelSettingsUI()
    window = LearningSettingsWidget()
    window.show()
    sys.exit(app.exec())
