#!/usr/bin/env python
import os
import sys
import imp
import argparse
import numpy as np

sys.path.append('.')
sys.path.append('./caffe-tensorflow')

from datasets.videodb import prepare_roidb, prepare_trajdb
from datasets.motdb import MotDB
from track_rcnn.minibatch import get_minibatch
from track_rcnn.train import train_net

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Track R-CNN network')

    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)

    parser.add_argument('--dir', dest='output_dir',
                        help='Output directory.',
                        default='tmp', type=str)

    parser.add_argument('--pretrain', dest='pretrain',
                        help='Pretrained model\'s checkpoint directory.',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Set GPU id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Choose videos
    train_videos = ['PETS09-S2L1']

    train_videos = ['ADL-Rundle-6',
                    'ETH-Bahnhof',
                    'ETH-Pedcross2',
                    'KITTI-13',
                    'PETS09-S2L1',
                    'TUD-Campus',
                    'Venice-2']

    # train_videos = ['ADL-Rundle-6',
    #                 'ADL-Rundle-8',
    #                 'ETH-Bahnhof',
    #                 'ETH-Pedcross2',
    #                 'ETH-Sunnyday',
    #                 'KITTI-13',
    #                 'KITTI-17',
    #                 'PETS09-S2L1',
    #                 'TUD-Campus',
    #                 'TUD-Stadtmitte', 
    #                 'Venice-2']

    # Load configuration
    path_config = os.path.join('track_rcnn', 'config.py')
    config = imp.load_source('*', path_config)
    cfg = config.cfg

    # Train set
    videodb = MotDB(train_videos)
    videodb.load_trajdb()

    train_trajdb = list(videodb.trajdb)
    # train_trajdb = [train_trajdb[4]]
    train_trajdb = prepare_trajdb(cfg, train_trajdb)
    train_roidb = prepare_roidb(videodb)
    print("#train_trajectories: %d" % len(train_trajdb))

    # Run trainining
    train_net(cfg, train_trajdb, train_roidb, args.output_dir, args.pretrain,
                max_iters=100000)

