#!/usr/bin/env python
import os
import sys
import argparse
import imp
import time
import numpy as np
import cv2
import cPickle
import h5py
import tensorflow as tf

sys.path.append('.')

from datasets.add_path import data_root
from datasets.motdb import MotDB
from track_rcnn.cnn import cnn_config
from track_rcnn.minibatch import prep_im_for_inputs

def get_conv5(cnn_model, ims):
    # Conv 1~5
    conv1_layers = cnn_model.conv1({'data': ims}, trainable=False)
    conv1 = conv1_layers.get_output()

    conv5_layers = cnn_model.conv5({'data': conv1}, trainable=False)
    conv5 = conv5_layers.get_output()

    return conv5

def load_cnn_params(sess, params_path):
    ignore_missing=True
    data_path = params_path
    data_dict = np.load(data_path).item()
    for key in data_dict:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                try:
                    var = tf.get_variable(subkey)
                    sess.run(var.assign(data))
                except ValueError:
                    if not ignore_missing:
                        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Extract the conv5 features.')

    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)

    parser.add_argument('--pretrain', dest='pretrain',
                        help='Pretrained model\'s checkpoint directory.',
                        default=None, type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Load configuration
    path_config = os.path.join('track_rcnn', 'config.py')
    config = imp.load_source('*', path_config)
    cfg = config.cfg

    # CNN model
    cnn_model = cnn_config[cfg.CNN_MODEL_NAME]

    # Training videos
    train_videos = ['ADL-Rundle-6',
                    'ETH-Bahnhof',
                    'ETH-Pedcross2',
                    'KITTI-13',
                    'PETS09-S2L1',
                    'TUD-Campus',
                    'Venice-2',

                    'ADL-Rundle-8',
                    'ETH-Sunnyday',
                    'KITTI-17',
                    'TUD-Stadtmitte']

    videodb = MotDB(train_videos)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Build CNN
        ims = tf.placeholder(tf.float32, [None, None, None, 3],
                                name='im_inputs')
        conv5 = get_conv5(cnn_model, ims)
        tf.initialize_all_variables().run()

        # Restore weights
        if args.pretrain == None:
            load_cnn_params(sess, cnn_model.params_path)
            print('Loaded CNN parameters from pretrained Caffe model.')
        else:
            pretrain_path = os.path.join('outputs', args.pretrain, 'model.ckpt')
            saver = tf.train.Saver()
            saver.restore(sess, pretrain_path)
            print('Loaded CNN parameters from pretrained Tensorflow model.')

        # Run
        conv5_root = os.path.join(data_root, 'conv5')
        if not os.path.exists(conv5_root):
            os.makedirs(conv5_root)

        for video_id, video_name in enumerate(videodb.name):
            print('Processing video [{}]'.format(video_name))
            n_steps = videodb.n_frames[video_id]
            path = os.path.join(videodb.data_path[video_id], 'conv5')
            im_scale = videodb.im_scale[video_id]
            if not os.path.exists(path):
                os.makedirs(path)

            conv5_path = os.path.join(conv5_root, '%s.h5' % video_name)
            with h5py.File(conv5_path, 'w') as hf:
                for t in range(n_steps):
                    # Prepare the data
                    start_time = time.time()
                    im_path = videodb.image_path_at(video_id, t)
                    # conv5_path = videodb.conv5_path_at(video_id, t)
                    im = cv2.imread(im_path)
                    im = prep_im_for_inputs(im, cfg.PIXEL_MEANS, im_scale)
                    ims_data = np.expand_dims(im, axis=0)

                    # Run CNN
                    conv5_data = sess.run(conv5, feed_dict={ims: ims_data})
                    data = conv5_data[0]

                    # Log and save features
                    print('video: {}, step: {} / {}, time: {}'.format(
                        video_name, t, n_steps, time.time()-start_time))
                    hf.create_dataset(str(t), data=data)



