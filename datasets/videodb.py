import sys
import os
import numpy as np
from utils.cython_bbox import bbox_overlaps

class VideoDB(object):
    """Video database."""

    def __init__(self, name):
        self._name = list(name)
        self._frame_shape = []
        self._n_frames = []
        self.division = division
        # For both trainset and test set:
        self._roidb = []
        # For trainset only:
        self._ids = []
        self._trajdb = []

    @ property
    def name(self):
        return self._name

    @ property
    def ids(self):
        return self._ids

    @ property
    def frame_shape(self):
        return self._frame_shape

    @ property
    def im_scale(self):
        return self._im_scale

    @ property
    def n_frames(self):
        return self._n_frames

    @ property
    def roidb(self):
        return self._roidb

    @ property
    def trajdb(self):
        return self._trajdb

    @ property
    def data_path(self):
        return self._data_path

    def image_path_at(self, video_id, t):
        raise NotImplementedError

    def create_rois_for_trajdb(self, cfg):
        for i, traj in enumerate(self.trajdb):
            for t, traj_t in enumerate(traj):
                # Get boxes
                boxes = [traj_t['gt_box']]
                n_samples = cfg.TRAIN.N_BOX_SAMPLES
                boxes += corrupt_boxes(traj_t['gt_box'], 0.5, n_samples)
                boxes = np.array(boxes)
                self._trajdb[i][t]['boxes'] = boxes

                # Compute overlaps
                gt_box = np.expand_dims(traj_t['gt_box'], 0)
                overlaps = bbox_overlaps(boxes.astype(np.float),
                                        gt_box.astype(np.float)).ravel()
                visible = self._trajdb[i][t]['visible']
                self._trajdb[i][t]['gt_overlaps'] = overlaps * visible

    def append_flipped_images(self):
        # TODO
        raise NotImplementedError

    def create_roidb_from_box_list(self, video_id, box_list, gt_roidb):
        name = self.name[video_id]
        n_frames = self.n_frames[video_id]
        n_ids = len(self.ids[video_id])

        roidb = []
        for fr in range(n_frames):
            boxes = box_list[fr]
            n_boxes = boxes.shape[0]
            overlaps = np.zeros((n_boxes), dtype=np.float32)

            gt_boxes = gt_roidb[fr]['boxes']
            if gt_boxes.size != 0:
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            gt_boxes.astype(np.float)) *\
                                np.tile(gt_roidb[fr]['visible'], (n_boxes, 1))
                overlaps = gt_overlaps
            
            roidb.append({'image': gt_roidb[fr]['image'],
                            'obj_ids': gt_roidb[fr]['obj_ids'],
                            'boxes': boxes,
                            'gt_overlaps': overlaps,
                            'flipped': False})
        return roidb

def merge_roidbs(a, b):
    assert len(a) == len(b)
    for i in xrange(len(a)):
        for t in xrange(len(a[i])):
            if a[i][t]['boxes'].shape[0] > 0:
                a[i][t]['boxes'] = np.vstack((a[i][t]['boxes'],
                                                b[i][t]['boxes']))
                a[i][t]['gt_overlaps'] = np.vstack((a[i][t]['gt_overlaps'],
                                                    b[i][t]['gt_overlaps']))
            else:
                a[i][t]['boxes'] = b[i][t]['boxes']
                a[i][t]['gt_overlaps'] = b[i][t]['gt_overlaps']
    return a

def rank_nearest_boxes(cbox, boxes, weight=np.array([[1, 1, 100, 100]])):
    """
    Rank the boxes according to their distance with the centroid box,
    return the ranking index.
    """
    n_boxes = boxes.shape[0]

    _cbox = np.zeros_like(cbox)
    _cbox[0] = (cbox[0] + cbox[2]) / 2
    _cbox[1] = (cbox[1] + cbox[3]) / 2
    _cbox[2] = (cbox[2] - cbox[0]) / 2
    _cbox[3] = (cbox[3] - cbox[1]) / 2

    _boxes = np.zeros_like(boxes)
    _boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    _boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    _boxes[:, 2] = (boxes[:, 2] - boxes[:, 0]) / 2
    _boxes[:, 3] = (boxes[:, 3] - boxes[:, 1]) / 2

    _cbox = np.tile(_cbox.reshape(1, -1), (n_boxes, 1))
    diff = _boxes - _cbox
    dists = np.linalg.norm(diff * weight, axis=1)
    rank_inds = np.argsort(dists)
    return rank_inds

def prepare_roidb(videodb):
    """ """
    gt_roidb = videodb.gt_roidb()
    ss_roidb = videodb.selective_search_roidb()
    roidb = merge_roidbs(gt_roidb, ss_roidb)
    return roidb

def prepare_trajdb(cfg, trajdb):
    """
    Prepare the trajdb for batch training.
    """
    n_steps = cfg.TRAIN.N_STEPS
    n_shift_steps = cfg.TRAIN.N_SHIFT_STEPS

    trajdb = list(trajdb)
    trajdb = _shift(trajdb, n_steps, n_shift_steps)
    return trajdb

def _shift(trajdb, n_steps, n_shift_steps):
    new_trajdb = []
    for i in range(len(trajdb)):
        s = 0
        while(s + 5 <= len(trajdb[i])): # TODO: minimum length
            if trajdb[i][s]['visible']:
                end = min(s+n_steps, len(trajdb[i]))
                new_trajdb.append(trajdb[i][s:end])
                s += n_shift_steps
            else:
                s += 1
    return new_trajdb




