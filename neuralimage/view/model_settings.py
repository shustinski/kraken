import os
import shutil
import tkinter as tk
from tkinter import filedialog

from PIL import Image

from UI.UI_design import *
from UI.my_ui_elements import *
from lib.config import Config


class ModelSettings(QWidget):
    log_data = pyqtSignal(object)
    samples_ready = pyqtSignal(object)
    gui_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setTitle("Настройка модели и обучения")
        self.main_grid = QVBoxLayout(self)

        self.area_widget = QWidget()
        self.scroll = QScrollArea(self.area_widget)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.scroll.setWidgetResizable(True)


        self.grid = QGridLayout(self.area_widget)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 100)

        y_grid_loc = 0

        jpg_path_text = QLabel("Путь до JPG")
        self.grid.addWidget(jpg_path_text, y_grid_loc, 0)

        self.jpg_path_label = ClickableLabel()
        self.jpg_path_label.setToolTip("Укажите путь к кадрам для выборки")
        self.jpg_path_label.clicked.connect(lambda: self.select_source())
        self.grid.addWidget(self.jpg_path_label, y_grid_loc, 1)
        y_grid_loc += 1

        cif_path_text = QLabel("Путь до CIF")
        self.grid.addWidget(cif_path_text, y_grid_loc, 0)

        self.cif_path_label = ClickableLabel()
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
        self.sample_prep_automat = SamplePrepSteps.convert_cif_thread
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
            case SamplePrepSteps.convert_cif_thread:
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

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = ModelSettings()
    window.show()
    sys.exit(app.exec())
