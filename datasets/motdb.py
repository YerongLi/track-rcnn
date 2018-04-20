import os
import glob
import imp
import numpy as np
from collections import OrderedDict
import cPickle

from add_path import data_root
from videodb import VideoDB
from utils.cython_bbox import bbox_overlaps

class MotConfig(object):
    """MOT Configuration."""
    def __init__(self, name, frame_shape, im_scale, n_frames, division, ids=[]):
        self.name = name
        self.frame_shape = frame_shape
        self.n_frames = n_frames
        self.division = division
        self.ids = ids
        self.im_scale = im_scale

    def __str__(self):
        return self.name.__name__

mot_configs = {\
        # # Format:
        # '':  MotConfig(name='',
        #                             frame_shape=(),
        #                             im_scale=,
        #                             n_frames=,
        #                             division='train',
        #                             ids=[]),
        'ADL-Rundle-6':  MotConfig(name='ADL-Rundle-6',
                                    frame_shape=(1920, 1080, 3),
                                    im_scale=1,
                                    n_frames=525,
                                    division='train',
                                    ids=range(24)),
        'ADL-Rundle-8':  MotConfig(name='ADL-Rundle-8',
                                    frame_shape=(1920, 1080, 3),
                                    im_scale=1,
                                    n_frames=654,
                                    division='train',
                                    ids=range(28)),
        'ETH-Bahnhof':  MotConfig(name='ETH-Bahnhof',
                                    frame_shape=(640, 480, 3),
                                    im_scale=3,
                                    n_frames=1000,
                                    division='train',
                                    ids=range(93)), # 171 on the website
        'ETH-Pedcross2':  MotConfig(name='ETH-Pedcross2',
                                    frame_shape=(640, 480, 4),
                                    im_scale=3,
                                    n_frames=837,
                                    division='train',
                                    ids=range(92)), # 133 on the website
        'ETH-Sunnyday':  MotConfig(name='ETH-Sunnyday',
                                    frame_shape=(640, 480, 3),
                                    im_scale=3,
                                    n_frames=354,
                                    division='train',
                                    ids=range(30)),
        'KITTI-13':  MotConfig(name='KITTI-13',
                                    frame_shape=(1242, 375, 3),
                                    im_scale=2.5,
                                    n_frames=340,
                                    division='train',
                                    ids=range(42)),
        'KITTI-17':  MotConfig(name='KITTI-17',
                                    frame_shape=(1224, 370, 3),
                                    im_scale=2.5,
                                    n_frames=145,
                                    division='train',
                                    ids=range(9)),
        'PETS09-S2L1':  MotConfig(name='PETS09-S2L1',
                                    frame_shape=(768, 576, 3),
                                    im_scale=2.5,
                                    n_frames=795,
                                    division='train',
                                    ids=range(19)),
        'TUD-Campus':  MotConfig(name='TUD-Campus',
                                    frame_shape=(640, 480, 3),
                                    im_scale=3,
                                    n_frames=71,
                                    division='train',
                                    ids=range(8)),
        'TUD-Stadtmitte':  MotConfig(name='TUD-Stadtmitte',
                                    frame_shape=(640, 480, 3),
                                    im_scale=3,
                                    n_frames=179,
                                    division='train',
                                    ids=range(10)),
        'Venice-2':  MotConfig(name='Venice-2',
                                    frame_shape=(1920, 1080, 3),
                                    im_scale=1,
                                    n_frames=600,
                                    division='train',
                                    ids=range(26))
        }

class MotDB(VideoDB):
    """MOT database."""

    def __init__(self, name):
        self._name = list(name)
        self._frame_shape = []
        self._im_scale = []
        self._n_frames = []
        # For both trainset and test set:
        self._roidb = []
        # For trainset only:
        self._ids = []
        self._trajdb = []

        # Data path:
        self._data_path = []

        for video_id, video_name in enumerate(self.name):
            mot_config = mot_configs[video_name]
            self._frame_shape.append(mot_config.frame_shape)
            self._im_scale.append(mot_config.im_scale)
            self._n_frames.append(mot_config.n_frames)
            self._ids.append(mot_config.ids)
            path = os.path.join(data_root, '2DMOT2015',
                                mot_config.division, video_name)
            self._data_path.append(path)

            assert os.path.exists(self._data_path[video_id]), \
                'Path does not exist: {}'.format(self._data_path[video_id])
    @ property
    def cache_path(self):
        cache_path = os.path.join(data_root, 'cache')
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def image_path_at(self, video_id, t):
        """
        Return the frame image of video[video_id] at time (t+1).
        Caution: Convert 0-based index to 1-based index.
        """
        return os.path.join(self._data_path[video_id], 'img1', '%06d.jpg'%(t+1))

    # def conv5_path_at(self, video_id):
    #     """Return the conv5 feature at time t."""
    #     # return os.path.join(self._data_path[video_id], 'conv5','%06d.pkl'%(t+1))
    #     return os.path.join(data_root,'conv5','%s.h5'%self.video_name[video_id])

    def load_trajdb(self):
        """
        Load all the trajdb's processed by Yu Xiang's matlab script from file.
        """
        self._trajdb = []
        for video_id, video_name in enumerate(self.name):
            for id_i, id in enumerate(self.ids[video_id]):
                traj = []
                path = os.path.join(self._data_path[video_id], 'traj',\
                                    'id%02d.txt' % (id + 1))
                with open(path, 'r') as f:
                    for line in f:
                        words = line.split(',')
                        fr = int(words[0]) - 1 # to 0-based index
                        obj_id = id + 1 # int(words[1])
                        x = float(words[2]) - 1 # to 0-based index
                        y = float(words[3]) - 1 # to 0-based index
                        w = float(words[4]) # to 0-based index
                        h = float(words[5]) # to 0-based index
                        conf = float(words[6])
                        occluded = int(words[7])
                        covered = float(words[8]) 
                        overlap = float(words[9])
                        area_inside = float(words[10])
                        # traj_t = OrderedDict() # Debug
                        traj_t = {}
                        traj_t['video_id'] = video_id
                        traj_t['video_name'] = video_name
                        traj_t['fr'] = fr
                        traj_t['obj_id'] = obj_id
                        traj_t['gt_box'] = np.array([x, y, x+w, y+h])
                        traj_t['visible'] = 1-occluded
                        traj_t['image'] = self.image_path_at(video_id, fr)
                        # traj_t['conv5'] = self.conv5_path_at(video_id)
                        traj_t['im_scale'] = self.im_scale[video_id]
                        traj_t['flipped'] = False
                        traj.append(traj_t)
                self._trajdb.append(traj)

    def gt_roidb(self):
        roidb = []
        for video_id, name in enumerate(self.name):
            cache_file = os.path.join(self.cache_path, name + '_gt_roidb.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    roidb_i = cPickle.load(fid)
                print '{} gt roidb loaded from {}'.format(name, cache_file)
            else:
                roidb_i = self._load_gt_roidb(video_id)
                with open(cache_file, 'wb') as fid:
                    cPickle.dump(roidb_i, fid, cPickle.HIGHEST_PROTOCOL)
                print 'wrote gt roidb to {}'.format(cache_file)
            roidb.append(roidb_i)
        return roidb

    def selective_search_roidb(self):
        gt_roidb = self.gt_roidb()
        roidb = []
        for i, name in enumerate(self.name):
            cache_file = os.path.join(self.cache_path,
                                        name + '_selective_search_roidb.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    roidb_i = cPickle.load(fid)
                print '{} ss roidb loaded from {}'.format(name, cache_file)
            else:
                roidb_i = self._load_selective_search_roidb(i, gt_roidb[i])
                with open(cache_file, 'wb') as fid:
                    cPickle.dump(roidb_i, fid, cPickle.HIGHEST_PROTOCOL)
                print 'wrote ss roidb to {}'.format(cache_file)
            roidb.append(roidb_i)
        return roidb

    def _load_gt_roidb(self, video_id):
        """
        Load all the trajdb's processed by Yu Xiang's matlab script from file.
        """
        n_frames = self.n_frames[video_id]
        roidb = [{
                'image': self.image_path_at(video_id, fr),
                'obj_ids': [],
                'boxes': [],
                'visible': [],
                'flipped': []
                }
                for fr in range(n_frames)]

        for id in self.ids[video_id]:
            filename = os.path.join(self._data_path[video_id], 'traj',\
                                'id%02d.txt' % (id+1))
            with open(filename, 'r') as f:
                for line in f:
                    words = line.split(',')
                    fr = int(words[0]) - 1 # to 0-based index
                    obj_id = int(words[1])
                    x = float(words[2]) - 1 # to 0-based index
                    y = float(words[3]) - 1 # to 0-based index
                    w = float(words[4]) # to 0-based index
                    h = float(words[5]) # to 0-based index
                    conf = float(words[6])
                    occluded = int(words[7])
                    covered = float(words[8]) 
                    overlap = float(words[9])
                    area_inside = float(words[10])

                    roidb[fr]['obj_ids'].append(id+1)
                    roidb[fr]['boxes'].append([x, y, x+w, y+h])
                    roidb[fr]['visible'].append(1 - occluded)
                    roidb[fr]['flipped'].append(False)

        for fr in range(n_frames):
            roidb[fr]['boxes'] = np.array(roidb[fr]['boxes']).astype(np.uint16)
            roidb[fr]['visible'] = np.array(roidb[fr]['visible'])
            roidb[fr]['flipped'] = np.array(roidb[fr]['flipped'])

            # Compute the overlap scores
            boxes = roidb[fr]['boxes']
            n_boxes = boxes.shape[0]
            overlaps = np.zeros((n_boxes), dtype=np.float32)

            gt_boxes = roidb[fr]['boxes']
            if gt_boxes.size != 0:
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            gt_boxes.astype(np.float)) *\
                                np.tile(roidb[fr]['visible'], (n_boxes, 1))
                overlaps = gt_overlaps

            roidb[fr]['gt_overlaps'] = overlaps

        return roidb

    def _load_selective_search_roidb(self, video_id, gt_roidb):
        name = self.name[video_id]
        filename = os.path.join(self.cache_path, '..', 'selective_search_data',
                                name + '.txt')
        assert os.path.exists(filename), \
                'Selective search data not found at: {}'.format(filename)

        n_frames = self.n_frames[video_id]
        box_list = [[] for _ in range(n_frames)]

        with open(filename, 'r') as f:
            for line in f:
                words = line.split(',')
                fr = int(words[0]) - 1 # to 0-based index
                x = float(words[2]) - 1 # to 0-based index
                y = float(words[3]) - 1 # to 0-based index
                w = float(words[4]) # to 0-based index
                h = float(words[5]) # to 0-based index
                box = np.array([x, y, x+w, y+h])
                box_list[fr].append(box)

        box_list = [np.array(box_list[fr]).astype(np.uint16)
                    for fr in range(n_frames)]
        return self.create_roidb_from_box_list(video_id, box_list, gt_roidb)

