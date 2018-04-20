#!/usr/bin/env python
import os
import sys
import glob
import imp
import numpy as np
import tensorflow as tf

sys.path.append('.')
sys.path.append('./caffe-tensorflow')

from datasets.videodb import VideoDB, prepare_trajdb, rank_nearest_boxes
from datasets.motdb import MotDB
from track_rcnn.minibatch import get_minibatch
from track_rcnn.layer import roi_pool
from kaffe.tensorflow import Network

def debug_motdb():
    train_videos = ['TUD-Stadtmitte',
                    'TUD-Campus',
                    'PETS09-S2L1',
                    'ETH-Bahnhof',
                    'ETH-Sunnyday',
                    'ETH-Pedcross2',
                    'ADL-Rundle-6',
                    'ADL-Rundle-8',
                    'KITTI-13',
                    'KITTI-17',
                    'Venice-2']

    test_videos = ['TUD-Crossing',
                    'PETS09-S2L2',
                    'ETH-Jelmoli',
                    'ETH-Linthescher',
                    'ETH-Crossing',
                    'AVG-TownCentre',
                    'ADL-Rundle-1',
                    'ADL-Rundle-3',
                    'KITTI-16',
                    'KITTI-19',
                    'Venice-1']
    # Debug
    path_config = os.path.join('track_rcnn', 'config.py')
    config = imp.load_source('*', path_config)
    cfg = config.cfg

    motdb = MotDB(train_videos)
    motdb.load_trajdb()

    # for traj in videodb.trajdb:
    #     print('\n')
    #     for track in traj:
    #         print(track)

    motdb.create_rois_for_trajdb(cfg)
    trajdb = prepare_trajdb(cfg, motdb.trajdb) # Prepare for batch training.
    inputs, targets = get_minibatch(cfg, trajdb[0:2])

    # print(videodb.roidb[100][0]['boxes'])

def test_max_pooling():
    """Understand the difference between 'VALID' and 'SAME' padding."""
    im_shape = [1925, 1080, 3]
    images = np.zeros([2]+im_shape)
    with tf.Graph().as_default(), tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, shape=[2]+im_shape,
                                        name='input_data')
        output = tf.nn.max_pool(input_data, [1, 16, 16, 1], [1, 16, 16, 1], 'VALID')
        out = sess.run(output, feed_dict={input_data: images})
        print(out.shape)

def debug_update_state():
    n_trajs = 2
    n_rois = 8

    with tf.Graph().as_default(), tf.Session() as sess:
        scores_t = tf.placeholder(tf.float32, shape=[n_rois])
        _scores_t = np.array([.1, .3, .7, .6, .1, .3, .4, .2])
        # _scores_t = np.array([.1, .3, .7, .6, .1, .3, .8, .2])


        feed_dict = {scores_t: _scores_t}

        # _choose_roi_inds
        rois_per_traj =  n_rois / n_trajs
        inds = []
        visible = []
        for i in range(n_trajs):
            scores_i_t = scores_t[i*rois_per_traj:(i+1)*rois_per_traj]
            max_score = tf.reduce_max(scores_i_t)
            visible.append(tf.greater(max_score, 0.5)) # TODO
            _idx = tf.to_float(tf.argmax(scores_i_t, 0))
            idx = tf.to_int32(_idx + i*rois_per_traj)
            inds.append(idx)
        roi_inds_t = tf.pack(inds)
        visible_t = tf.pack(visible)

        _roi_inds_t, _visible_t = sess.run([roi_inds_t, visible_t], feed_dict=feed_dict)
        print(_roi_inds_t)
        print(_visible_t)

        # #
        rnn_size = 11
        state_t = tf.placeholder(tf.float32, shape=[n_rois, 11])
        state_tm1 = tf.placeholder(tf.float32, shape=[n_rois, 11])
        _state_t = np.tile(np.reshape(np.arange(n_rois), (n_rois, 1)), (1, 11))
        _state_tm1 = np.tile(np.reshape(np.arange(n_rois), (n_rois, 1)), (1, 11)) + .1
        print(_state_t)
        print(_state_tm1)

        feed_dict = {roi_inds_t: _roi_inds_t, 
                    visible_t: _visible_t, 
                    state_t: _state_t,
                    state_tm1: _state_tm1}

        rois_per_traj =  n_rois / n_trajs
        indices = tf.reshape(tf.tile(tf.expand_dims(tf.to_int32(roi_inds_t), 1),
                                        [1, rois_per_traj]), [-1])
        cond_t = tf.tile(tf.reshape(tf.tile(tf.expand_dims(visible_t, 1),
                            [1, rois_per_traj]), [n_rois, 1]), [1, rnn_size])
        print('indices', indices.get_shape())
        print('shape', cond_t.get_shape())
        print('shape', state_t.get_shape())
        print('shape', state_tm1.get_shape())

        state_out = tf.select(cond_t,
                                tf.gather(state_t, indices),
                                state_tm1)

        _indices,  _state_out = sess.run([indices, state_out], feed_dict=feed_dict)
        print(_indices)
        print(_state_out)


def debug_choose_score():
    n_rois = 8
    n_trajs = 2
    n_steps = 3
    with tf.Graph().as_default(), tf.Session() as sess:
        scores = tf.placeholder(tf.float32, shape=[n_rois, n_steps])
        roi_inds = tf.placeholder(tf.int32, shape=[n_trajs, n_steps])

        _scores = np.reshape(np.arange(n_rois * n_steps), (n_rois, n_steps))
        _roi_inds = np.array([[1, 2, 3], [4, 4, 4]], dtype='int32')

        chosen_scores = []
        for t in range(n_steps):
            chosen_scores.append(tf.expand_dims(
                                tf.gather(scores[:, t], roi_inds[:, t]), dim=1))
        chosen_scores = tf.concat(1, chosen_scores)
        print('chosen scores:', chosen_scores.get_shape())

        feed_dict = {scores:_scores, roi_inds:_roi_inds}
        print(_scores)
        print(_roi_inds)
        print(sess.run(chosen_scores, feed_dict=feed_dict))

def debug_roi_pool_grad():
    """
    input_data: n_data X height_ X width_ X channel
                (e.g. 2 X 120 X 67 X 512)
    input_rois: n_rois X 5
                (e.g. 128 X 5)
    """
    input_data = tf.placeholder(tf.float32, shape=[2, 225, 225, 5])
    input_rois = tf.placeholder(tf.float32, shape=[14, 5])
    pool5 = roi_pool(input_data, input_rois,
                spatial_scale=1/16., pooled_h=6, pooled_w_=6
    return

if __name__ == '__main__':
    # debug_update_state()
    debug_choose_score()
