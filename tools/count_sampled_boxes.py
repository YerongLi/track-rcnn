#!/usr/bin/env python
"""
This code is used  to choose the threshold of RoIs sampling in each minibatch.
"""
import os
import sys
import glob
import imp
import numpy as np

sys.path.append('.')

from datasets.videodb import prepare_trajdb
from datasets.motdb import MotDB

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

path_config = os.path.join('track_rcnn', 'config.py')
config = imp.load_source('*', path_config)
cfg = config.cfg

pos_th = 0.9

for video in train_videos:
    motdb = MotDB([video])
    motdb.load_trajdb()
    motdb.create_rois_for_trajdb(cfg)

    tot_cnt = 0
    pos_cnt = 0
    for traj in motdb.trajdb:
        for traj_t in traj:
            tot_cnt += traj_t['overlaps'].shape[0]
            pos_cnt += np.sum(traj_t['overlaps'] >= pos_th)
    neg_cnt = tot_cnt - pos_cnt

    print("video: %s, pos %.3f, neg %.3f"\
            % (video, float(pos_cnt) / tot_cnt, float(neg_cnt) / tot_cnt))

# hi_th = 0.9
# lo_th = 0.1
# 
# for video in train_videos:
#     motdb = MotDB([video])
#     motdb.load_trajdb()
#     motdb.create_rois_for_trajdb(cfg)
# 
#     tot_cnt = 0
#     hi_cnt = 0
#     lo_cnt = 0
#     for traj in motdb.trajdb:
#         for traj_t in traj:
#             tot_cnt += traj_t['overlaps'].shape[0]
#             hi_cnt += np.sum(traj_t['overlaps'] > hi_th)
#             lo_cnt += np.sum(traj_t['overlaps'] < lo_th)
#     mid_cnt = tot_cnt - hi_cnt - lo_cnt
# 
#     print("video: %s, hi %.3f, mid %.3f, lo %.3f"\
#             % (video, float(hi_cnt) / tot_cnt, float(mid_cnt) / tot_cnt,
#             float(lo_cnt) / tot_cnt))


