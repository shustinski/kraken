from pathlib import Path

from torch.fx.experimental.unification.multipledispatch.dispatcher import source

from lib.data_interfaces import SampleGenerationSettings, TrainingParameters, SampleCutMode, SamplePrepareSettings, \
    WorkMode, RecognitionParameters
from lib.file_func import filter_images
from lib.message_bus import MessageBus
from model.general_neural_handler import GeneralNeuralHandler


def logger(msg:str):
    print(msg)

def question(theme:str, message:str) -> bool:
    while True:
        print(theme)
        print(message)
        answer  = input('Y/N, Д/Н')
        if answer.upper() in 'YД':
            return True
        elif  answer.upper() in 'NН':
            return False
        else:
            print('Невозможно определить ответ. Попробуйте ещё раз')

def test_train_and_recognition():
    image_path = Path('D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/jpg')
    label_path = Path('D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/bin')
    work_mode = WorkMode.train_and_recognition
    param_prep = SamplePrepareSettings()
    params_cut = SampleGenerationSettings(step=16,
                                          segment_size=(16,16),
                                          vertical_rotation=False,
                                          horizontal_rotation=False,
                                          channels=1)
    training_settings = TrainingParameters(image_path=image_path,
                                           label_path=label_path,
                                           shuffle=False,
                                           validation=False,
                                           validation_percent=20,
                                           batch_size=4,
                                           cut_mode=SampleCutMode.online,
                                           colors=1,
                                           epochs=5,
                                           generation=params_cut,
                                           prepare=param_prep)
    recognition_parameters = RecognitionParameters(
        source_files=filter_images(Path('D:/MSP/NN/M1/Source/M1_BS')),
        result_folder=Path('D:/MSP/NN/M1/Source/result_test_new'),
        model='SmallTransformer',
        batch_size=8,
        overlap=4,
        part_size=(16,16)
    )
    message_bus = MessageBus()
    message_bus.subscribe('logging', logger)
    message_bus.subscribe('training', logger)

    neuaral_handler = GeneralNeuralHandler(work_mode=work_mode,
                                           recogniton_parameters=recognition_parameters,
                                           tranining_parameters=training_settings,
                                           question_module=question,
                                           message_bus=message_bus)
    neuaral_handler.start()

def test_recognition_only():
    image_path = Path('D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/jpg')
    label_path = Path('D:/MSP/NN/M1/SAMPLES/try_a_lot_of_frames/bin')
    model_path = Path('D:/NN/octal/m1/OCTAL_M1 для Нади/Wellnet_shift_4_rot90_rot180_cif_epoch40.pth')
    work_mode = WorkMode.recognition_only
    source_files = filter_images(Path('D:/MSP/NN/M1/Source/M1_BS'))
    param_prep = SamplePrepareSettings()
    recognition_parameters = RecognitionParameters(
        source_files=source_files,
        result_folder=Path('D:/MSP/NN/M1/Source/result_test_new'),
        model=model_path,
        part_size=(256,256),
        batch_size=8,
        overlap=8
    )
    message_bus = MessageBus()
    message_bus.subscribe('logging', logger)
    message_bus.subscribe('training', logger)

    neuaral_handler = GeneralNeuralHandler(work_mode=work_mode,
                                           recogniton_parameters=recognition_parameters,
                                           question_module=question,
                                           message_bus=message_bus)
    neuaral_handler.start()