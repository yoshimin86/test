# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

import i3d

_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

FLAGS = tf.flags.FLAGS

#tf.flags.DEFINE_string('eval_type', 'rgb', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

def make(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type

  print('eval_type :', eval_type)

  imagenet_pretrained = FLAGS.imagenet_pretrained

  # num of classes in your dataset
  nActions=5
  # num of classes in Kinetics
  NUM_CLASSES = 400
  if eval_type == 'rgb600':
    NUM_CLASSES = 600

  if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

  if eval_type == 'rgb600':
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
  else:
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'rgb600', 'joint']:
    print('Building RGB models...', end='')
    # Define RGB input
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    my_rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

    # Build RGB model
    original_rgb_name = 'RGB'
    with tf.variable_scope(original_rgb_name):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)

    my_rgb_name = 'my_RGB'
    with tf.variable_scope(my_rgb_name):
      my_rgb_model = i3d.InceptionI3d(
          nActions, spatial_squeeze=True, final_endpoint='Logits')
      my_rgb_logits, _ = my_rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)

    # Define variable names in RGB model
    rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == original_rgb_name:
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len(original_rgb_name+'/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    my_rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == my_rgb_name:
        if eval_type == 'rgb600':
          my_rgb_variable_map[variable.name.replace(':0', '')[len(my_rgb_name+'/inception_i3d/'):]] = variable
        else:
          my_rgb_variable_map[variable.name.replace(':0', '')] = variable

    # Make saver for loading or saving weights(.ckpt)
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    my_rgb_saver = tf.train.Saver(var_list=my_rgb_variable_map, reshape=True)
    print('Done!')

  if eval_type in ['flow', 'joint']:
    print('Building Flow models...', end='')
    # Define Flow input
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    my_flow_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))

    # Build Flow model
    original_flow_name = 'Flow'
    with tf.variable_scope(original_flow_name):
      flow_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      flow_logits, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)

    my_flow_name = 'my_Flow'
    with tf.variable_scope(my_flow_name):
      my_flow_model = i3d.InceptionI3d(
          nActions, spatial_squeeze=True, final_endpoint='Logits')
      my_flow_logits, _ = my_flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)

    # Define variable names in Flow model
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == original_flow_name:
        flow_variable_map[variable.name.replace(':0', '')] = variable

    my_flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == my_flow_name:
        my_flow_variable_map[variable.name.replace(':0', '')] = variable

    # Make saver for loading or saving weights(.ckpt)
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
    my_flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
    print('Done!')


  # Define the last output of model
  if eval_type == 'rgb' or eval_type == 'rgb600':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)

  # Define the last output of model
  if eval_type == 'rgb' or eval_type == 'rgb600':
    my_model_logits = my_rgb_logits
  elif eval_type == 'flow':
    my_model_logits = my_flow_logits
  else:
    my_model_logits = my_rgb_logits + my_flow_logits
  my_model_predictions = tf.nn.softmax(my_model_logits)

  with tf.Session() as sess:
    feed_dict = {}
    sess.run(tf.global_variables_initializer())

    # Load pretrained weights
    if eval_type in ['rgb', 'rgb600', 'joint']:
      print('Loading RGB weights...', end='')
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
      rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
      feed_dict[rgb_input] = rgb_sample
      print('Done!')

    if eval_type in ['flow', 'joint']:
      print('Loading Flow weights...', end='')
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      flow_sample = np.load(_SAMPLE_PATHS['flow'])
      feed_dict[flow_input] = flow_sample

    if eval_type in ['rgb', 'rgb600', 'joint']:
      # Copy weights in original model to your one
      print('Copying RGB weights...', end='')
      for key, variable in rgb_variable_map.items():
        if key==original_rgb_name+"/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/w":
          break
        my_rgb_variable_map[key.replace(original_rgb_name, my_rgb_name)]=variable
      print('Done!')

    if eval_type in ['flow', 'joint']:
      print('Copying Flow weights...', end='')
      # Copy weights in original model to your one
      for key, variable in flow_variable_map.items():
        if key==original_flow_name+"/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/w":
          break
        my_rgb_variable_map[key.replace(original_flow_name, my_flow_name)]=variable
      print('Done!')

    save_path = './weights/'
    os.makedirs(save_path, exist_ok=True)
    if eval_type in ['rgb', 'rgb600', 'joint']:
      print('\nShape of the last layer of RGB model')
      print(sess.run(rgb_variable_map[original_rgb_name+"/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/w"]).shape)
      print(sess.run(my_rgb_variable_map[my_rgb_name+"/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/w"]).shape)
      rgb_save_path = save_path+my_rgb_name+"_model.ckpt"
      my_rgb_saver.save(sess, rgb_save_path)
      print(my_rgb_name, 'model weights were saved in', rgb_save_path)

    if eval_type in ['flow', 'joint']:
      print('\nShape of the last layer of Flow model')
      print(sess.run(flow_variable_map[original_flow_name+"/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/w"]).shape)
      print(sess.run(my_flow_variable_map[my_flow_name+"/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/w"]).shape)
      flow_save_path = save_path+my_flow_name+"_model.ckpt"
      my_rgb_saver.save(sess, flow_save_path)
      print(my_flow_name, 'model weights were saved in', flow_save_path)

    print('\nEvaluating sample video...')
    pred, my_pred = sess.run([model_predictions, my_model_predictions], feed_dict=feed_dict)
    pred, my_pred = pred[0], my_pred[0]
    pred, my_pred = np.argsort(pred)[::-1], np.argsort(my_pred)[::-1]
    print('Top 3 classes')
    print('Original model prediction:', pred[:3])
    print('  Your   model prediction:', my_pred[:3])


if __name__ == '__main__':
  tf.app.run(make)
