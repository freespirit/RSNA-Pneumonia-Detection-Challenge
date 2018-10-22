import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from RetinaNet import retinanet, retina_loss, predict_box

FILE_TRAIN_IMAGES = "data/stage_1_train_images.zip"
FILE_TRAIN_LABELS = "data/stage_1_train_labels.zip"
FILE_TEST_IMAGES = "data/stage_1_test_images.zip"

TMP_DATASET_DIR = "/tmp/rsna_dataset"
TMP_DIR_TRAIN = "train"
TMP_DIR_TEST = "test"

EPOCHS = 1
BATCH_SIZE = 40


def extract_data(data_filename, dir="data"):
    """Read an archive containing challenge's DICOM images and extract in target directory
    Returns the directory to which the files were extracted
    :rtype String
    """
    target_dir = os.path.join(TMP_DATASET_DIR, dir)

    with ZipFile(data_filename) as dataset:
        dataset.extractall(path=target_dir)
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


def generate_kaggle_test_data(images):
    for (patientId, pixels) in images:
        yield (patientId, pixels)


# TODO consider image augmentation as in the ResNet paper section 3.4 Implementation
# TODO use weight decay of 0.0001 and momentum of 0.9


def main(argv):
    train_dataset = make_train_dataset()
    iterator = train_dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    next_batch = iterator.get_next()

    pixels = tf.placeholder(dtype=tf.float32, shape=[None, 1024, 1024, 1])
    target_class = tf.placeholder(dtype=tf.uint8, shape=[None, 1], name="classs")
    target_boxes = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="anchor_box")

    inputs = pixels
    inputs = tf.image.resize_images(inputs, [640, 640], method=ResizeMethod.NEAREST_NEIGHBOR)
    inputs = tf.image.grayscale_to_rgb(inputs)
    outputs = retinanet(inputs, 1)
    loss = retina_loss(outputs, target_class, target_boxes)

    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter("/tmp/rsna_model", sess.graph)
        writer.close()

        for epoch in range(EPOCHS):
            print(f"EPOCH_{epoch}: ")
            sess.run(iterator_init_op)
            batch_index = 1
            while True:
                try:
                    batch = sess.run(next_batch)
                    (x, y, width, height, label) = batch[1]
                    batch_pixels = np.array(batch[0]).reshape(-1, 1024, 1024, 1)
                    batch_classes = np.array(label).reshape(-1, 1)
                    batch_boxes = np.array([x, y, width, height]).reshape(-1, 4)

                    sess.run(minimize, feed_dict={pixels: batch_pixels,
                                                  target_boxes: batch_boxes,
                                                  target_class: batch_classes})

                    print(f"{epoch} -> {batch_index}")
                    batch_index += 1
                    # break # TODO TMP
                except tf.errors.OutOfRangeError:
                    print(f"Out of range with count: {batch_index}")
                    break

    kaggle_test_dataset = make_kaggle_dataset()
    kaggle_iterator = kaggle_test_dataset.make_initializable_iterator()
    kaggle_iterator_init_op = kaggle_iterator.initializer
    next_batch = kaggle_iterator.get_next()
    kaggle_predictions = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(kaggle_iterator_init_op)
        batch_index = 1
        while True:
            try:
                print(f"Processing batch {batch_index}...")
                batch = sess.run(next_batch)
                patient_id = batch[0]
                batch_pixels = np.array(batch[1]).reshape(-1, 1024, 1024, 1)

                probability, box = sess.run(predict_box(outputs), feed_dict={pixels: batch_pixels})

                df_prob = pd.DataFrame(probability, columns=['probability'])
                df_box = pd.DataFrame(box, columns=['x', 'y', 'width', 'height'])
                df_prediction = pd.concat([df_prob, df_box], axis=1)
                df = pd.DataFrame(patient_id, columns=['patientId'])
                df['PredictionString'] = df_prediction.iloc[:, :].apply(lambda x: " ".join(x.map('{:.4f}'.format)), axis=1)
                kaggle_predictions.append(df)

                total_processed = sess.run(batch_index * tf.shape(patient_id)[0])
                print(f"Total test images processed: {total_processed}")
                batch_index += 1

                # break # TODO TMP
            except tf.errors.OutOfRangeError:
                break

    df = pd.concat(kaggle_predictions)
    print(df)
    df.to_csv("data/submission.csv", index=False)


def make_train_dataset():
    train_labels = pd.read_csv("data/stage_1_train_labels.csv")

    train_data_dir = extract_data(FILE_TRAIN_IMAGES, TMP_DIR_TRAIN)
    train_data_generator = generate_training_data(read_images(train_data_dir), train_labels)

    train_dtype = (tf.uint8, (tf.float32, tf.float32, tf.float32, tf.float32, tf.uint8))
    return tf.data.Dataset \
        .from_generator(lambda: train_data_generator, train_dtype) \
        .batch(BATCH_SIZE)


def make_kaggle_dataset():
    kaggle_data_dir = extract_data(FILE_TEST_IMAGES, TMP_DIR_TEST)
    kaggle_data_generator = generate_kaggle_test_data(read_images(kaggle_data_dir))

    kaggle_dtype = (tf.string, tf.uint8)
    return tf.data.Dataset \
        .from_generator(lambda: kaggle_data_generator, kaggle_dtype) \
        .batch(BATCH_SIZE)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
