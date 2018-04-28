# Source: https://github.com/datitran/raccoon_dataset.git
"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record --label_map_file=label_map.pbtxt

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record --label_map_file=label_map.pbtxt
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
import io
import pandas as pd
import random
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_file', '', 'Path to label map file')
FLAGS = flags.FLAGS


def parse_label_map_file(fname):
    names = []
    ids = []
    with open(fname) as f:
        lines = f.readlines()
    for line in lines:
        if "id:" in line:
            ids.append(int(line.split("id:")[-1].strip()))
        elif "name:" in line:
            names.append(line.split("name:")[-1].strip().replace("'", ""))
    return {name:id for (name, id) in zip(names, ids)}


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, label_map):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(np.clip(row['xmin'] / width, 0, 1))
        xmaxs.append(np.clip(row['xmax'] / width, 0, 1))
        ymins.append(np.clip(row['ymin'] / height, 0, 1))
        ymaxs.append(np.clip(row['ymax'] / height, 0, 1))
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), "/".join(FLAGS.csv_input.split("/")[:-1]))
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    label_map = parse_label_map_file(FLAGS.label_map_file)
    examples_list = [create_tf_example(group, path, label_map) for group in grouped]
    random.shuffle(examples_list)
    for example in examples_list:
        writer.write(example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()

