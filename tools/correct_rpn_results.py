#!/usr/bin/env python
"""
Used to add ',' to the selective search results.
"""
import os
import numpy

train_videos = ['ADL-Rundle-6',
                'ADL-Rundle-8',
                'ETH-Bahnhof',
                'ETH-Pedcross2',
                'ETH-Sunnyday',
                'KITTI-13',
                'KITTI-17',
                'PETS09-S2L1',
                'TUD-Campus',
                'TUD-Stadtmitte', 
                'Venice-2']

src_root = 'data/_selective_search_data'
dst_root = 'data/selective_search_data'

for video in train_videos:
    print('Processing %s...' % video)
    src_path = os.path.join(src_root, video + '.txt')
    dst_path = os.path.join(dst_root, video + '.txt')

    new_lines = []
    with open(src_path, 'r') as fin:
        for line in fin:
            words = line.split(' ')
            new_line = words[0]
            i = 1
            for word in words[1:]:
                if i in [2, 3]:
                    word = str(float(word) + 1)
                new_line += ', ' + word
                i += 1
            new_lines.append(new_line)

    with open(dst_path, 'w') as fout:
        for new_line in new_lines:
            fout.write(new_line)

