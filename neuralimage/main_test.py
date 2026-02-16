from tests.general_test import test_train_and_recognition, test_recognition_only
from tests.image_test import run_image_tests
from tests.test_dataset import run_dataset_test




def main():
    # run_image_tests()
    # run_dataset_test()
    # test_train_and_recognition()
    test_recognition_only()

if __name__ == '__main__':
    main()
