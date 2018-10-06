import tensorflow as tf

from zipfile import ZipFile
import pydicom

import numpy as np
import pandas as pd

import os

FILE_TRAIN_IMAGES = "data/stage_1_train_images.zip"
FILE_TRAIN_LABELS = "data/stage_1_train_labels.zip"
FILE_TEST_IMAGES = "data/stage_1_test_images.zip"

TMP_DATASET_DIR = "/tmp/rsna_dataset"
TMP_DIR_TRAIN = "train"
TMP_DIR_TEST = "test"

EPOCHS = 1
BATCH_SIZE = 500


def extract_data(data_filename, dir="data"):
    """Read an archive containing challenge's DICOM images and extract in target directory
    Returns the directory to which the files were extracted
    :rtype String
    """
    target_dir = os.path.join(TMP_DATASET_DIR, dir)

    # with ZipFile(data_filename) as dataset:
    #     dataset.extractall(path=target_dir)
    print("INFO: RSNA dataset extracted to tmp dir ", target_dir)

    return target_dir


def read_images(target_dir):
    """Read a directory containing challenge's DICOM images and yield the image's name and pixels.
    :rtype generator
    """
    for file_name in os.listdir(target_dir):
        full_name = os.path.join(target_dir, file_name)
        with pydicom.dcmread(full_name) as dcm:
            id = file_name.replace(".dcm", "")
            # print(f"reading {id}")
            yield (f"{id}", dcm.pixel_array)


def generate_training_data(images, labels):
    for (patientId, pixels) in images:
        patient_entries = labels.loc[labels['patientId'] == patientId]
        # print("patient_entries: ", patient_entries)
        for index, entry in patient_entries.iterrows():
            # print("entry: ", entry)
            x = entry['x']
            y = entry['y']
            width = entry['width']
            height = entry['height']
            label = entry['Target']

            yield (pixels, (x, y, width, height, label))


def generate_test_data(images):
    for (patientId, pixels) in images:
        yield pixels


def main(argv):
    # entries = np.array([entry for entry in test_generator])
    batch_size = tf.placeholder
    train_labels = pd.read_csv("data/stage_1_train_labels.csv")

    train_data_dir = extract_data(FILE_TRAIN_IMAGES, TMP_DIR_TRAIN)
    train_data_generator = generate_training_data(read_images(train_data_dir), train_labels)

    train_dtype = ((tf.uint8), (tf.float32, tf.float16, tf.float16, tf.float16, tf.uint8))
    train_dataset = tf.data.Dataset.from_generator(lambda: train_data_generator, train_dtype)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    iterator_init_op = iterator.make_initializer(train_dataset)
    next = iterator.get_next()

    with tf.Session() as sess:
        for epoch in range(EPOCHS):
            sess.run(iterator_init_op)
            print(f"{epoch} -> ", sess.run(next))
            try:
                sess.run(next)
            except tf.errors.OutOfRangeError:
                pass

    test_dtype = (tf.uint8)
    test_data_dir = extract_data(FILE_TEST_IMAGES, TMP_DIR_TEST)
    test_data_generator = generate_test_data(read_images(test_data_dir))
    test_dataset = tf.data.Dataset.from_generator(lambda: test_data_generator, test_dtype)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
