#!/usr/bin/env python
import os
import sys
import imp
import argparse
import numpy as np
import cv2 # Need to keep this for a tensorflow bug
import tensorflow as tf

sys.path.append('.')

from datasets.videodb import prepare_roidb, prepare_trajdb
from datasets.motdb import MotDB
from track_rcnn.net import TrackRCNN
from track_rcnn.test import track_obj

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Track R-CNN network')

    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)

    parser.add_argument('--pretrain', dest='pretrain',
                        help='Pretrained model\'s checkpoint directory.',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # print(os.path.dirname(os.path.realpath(__file__)))
    # path = '/cvgl/u/kuanfang/track-rcnn/data/2DMOT2015/train/ETH-Sunnyday/img1/000001.jpg'
    # im = cv2.imread(path)
    # print(path, im)

    args = parse_args()

    # Set GPU id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Choose videos
    valid_videos = ['ETH-Sunnyday']
    valid_videos = ['ADL-Rundle-8']

    # valid_videos = ["ADL-Rundle-6",
    #                 "ETH-Bahnhof",
    #                 "ETH-Sunnyday",
    #                 "KITTI-13",
    #                 "PETS09-S2L1",
    #                 "TUD-Campus",
    #                 "Venice-2"]

    # Load configuration
    path_config = os.path.join('track_rcnn', 'config.py')
    config = imp.load_source('*', path_config)
    cfg = config.cfg

    # Train set
    videodb = MotDB(valid_videos)
    videodb.load_trajdb()

    valid_trajdb = list(videodb.trajdb)
    # valid_trajdb = [videodb.trajdb[0]]
    valid_roidb = prepare_roidb(videodb)
    print("#valid_trajectories: %d" % len(valid_trajdb))


    with tf.Graph().as_default(), tf.Session() as sess:
        # Build 
        m = TrackRCNN(cfg, mode='test')

        # Retore pretrained model
        saver = tf.train.Saver()
        pretrain_path = os.path.join('outputs', args.pretrain, 'model.ckpt')
        saver.restore(sess, pretrain_path)
        print('Loaded model from the checkpoint.')

        # Prepare data
        ind = 1
        trajdb = valid_trajdb[ind]
        video_id = trajdb[0]['video_id']
        roidb = valid_roidb[video_id]

        # Run validining
        track_obj(cfg, sess, m, trajdb, roidb)
