from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import hashlib
import io
import os
import json

import numpy as np
import PIL.Image as pil
import tensorflow as tf

#from object_detection.utils import dataset_util
#from object_detection.utils import label_map_util
#from object_detection.utils.np_box_ops import iou

import pandas as pd 

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


tf.app.flags.DEFINE_string('image_dir', '', 'Location of images ')
tf.app.flags.DEFINE_string('output_path', '', 'Path to which TFRecord files wiil be written')
tf.app.flags.DEFINE_string('csv_file', '', 'Path of csv file')
tf.app.flags.DEFINE_integer('validation_set_size', '500', 'Number of images to be used as a validation set.')
tf.app.flags.DEFINE_string('resize', '','resize the image to aim size "width, height" like "200,3000"')
tf.app.flags.DEFINE_string('labels_map_path', '','Path to labels mapping i.e profile.train.json')

FLAGS = tf.app.flags.FLAGS


def convert_csv_to_tfrecords(image_dir, output_path, labels_map_path, csv_file, validation_set_size):
  with open(labels_map_path) as f:
    labels_map = json.load(f)
  train_count = 0
  val_count = 0

  train_writer = tf.python_io.TFRecordWriter('%s_train.tfrecord'%
                                             output_path)
  val_writer = tf.python_io.TFRecordWriter('%s_val.tfrecord'%
                                           output_path)
  data = pd.read_csv(csv_file)
  data = data.fillna(0) 

  img_num = 0
  for i in range(len(data)):
    #print(data.keys())
    img_id = data['itemid'][i]
    img_path = data['image_path'][i]
    if img_path.endswith('.jpg'):
      pass
    else:
      img_path = img_path + '.jpg'
    print("Image Name: " + str(img_id))
    print("Image Path: " + img_path)
    is_validation_img = img_num < validation_set_size
    img_num += 1
    label = {}
    for key in labels_map.keys():
      label[key] = data[key][i]
    #example = prepare_example(img_path, label, box)
    #example = prepare_example_1(img_path, tmp)
    example = prepare_example_2(img_id, img_path, label)
    if is_validation_img:
      val_writer.write(example.SerializeToString())
      val_count += 1
    else:
      train_writer.write(example.SerializeToString())
      train_count += 1

  train_writer.close()
  val_writer.close()

def prepare_example_2(image_id, image_path, labels):
  #image_id = os.path.basename(image_path)
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = pil.open(encoded_png_io)
  
  key = hashlib.sha256(encoded_png).hexdigest()
  width,height = image.size

  img_byte_arr = io.BytesIO()
  image.save(img_byte_arr,format='PNG',quality=100)
  encoded_png = img_byte_arr.getvalue()
  feature_map = {
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/filepath': bytes_feature(image_path.encode('utf8')),
    'image/source_id': int64_feature(image_id),
    'image/key/sha256': bytes_feature(key.encode('utf8')),
    'image/encoded': bytes_feature(encoded_png),
    'image/format': bytes_feature('png'.encode('utf8')),
  }
  for key in labels:
    feature_map[key] = float_feature(labels[key])
      
  example = tf.train.Example(features=tf.train.Features(feature=feature_map))
  return example

def main(_):
  convert_csv_to_tfrecords(
      image_dir=FLAGS.image_dir,
      output_path=FLAGS.output_path,
      labels_map_path=FLAGS.labels_map_path,
      csv_file=FLAGS.csv_file,
      validation_set_size=FLAGS.validation_set_size)

if __name__ == '__main__':
  tf.app.run()
