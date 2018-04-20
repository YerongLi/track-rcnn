import os 
import time
import numpy as np
import h5py
import cv2

from datasets.add_path import data_root
from datasets.videodb import rank_nearest_boxes
# from motion.motion_prior import MotionPrior, ConstVelMP, RNNMP

def get_minibatch(cfg, trajdb, roidb, dtype='image', random=True):
    """Given a trajdb(trajectory batch), construct a minibatch sample."""
    n_steps = len(trajdb)
    n_rois = cfg.TRAIN.N_ROIS
    n_pos_rois = np.round(cfg.TRAIN.POS_FRACTION * n_rois).astype('int32')
    n_neg_rois = np.round(cfg.TRAIN.NEG_FRACTION * n_rois).astype('int32')

    # Get the input images, preprcess
    if dtype == 'image':
        im_inputs, im_scales = get_input_images(cfg, trajdb)
    elif dtype == 'conv5':
        im_inputs, im_scales = get_input_conv5s(cfg, trajdb)

    # Get the input RoI and ground truth scores
    roi_inputs = np.zeros((n_steps, n_rois, 5), dtype=np.float32)
    score_targets = np.zeros((n_steps, n_rois), dtype=np.float32)

    for t in range(n_steps):
        fr = trajdb[t]['fr']

        if t == 0:
            insert_gt = True
        else:
            insert_gt = False

        im_rois, overlaps = _sample_rois(cfg, trajdb[t], roidb[fr],
                            n_rois, n_pos_rois, n_neg_rois, random, insert_gt)

        if t == 0:
            best_ind = overlaps.argmax()
            im_rois[:, :] = im_rois[best_ind, :]
            overlaps[:] = overlaps[best_ind]

        # Add to RoIs inputs
        rois = _project_im_rois(im_rois, im_scales[t])
        batch_ind = np.zeros((rois.shape[0], 1))
        rois_this_image = np.hstack((batch_ind, rois))
        roi_inputs[t, :, :] = rois_this_image

        # Add to targets
        scores = overlaps
        # scores = (overlaps >= cfg.TRAIN.POS_TH)
        score_targets[t, :] = scores.astype('float32')

    batch_data = {'im_inputs': im_inputs, 
                'roi_inputs': roi_inputs,
                'score_targets': score_targets}
    return batch_data

def im_list_to_inputs(ims):
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    n_steps = len(ims)
    im_inputs = np.zeros((n_steps, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
    for t in xrange(n_steps):
        im = ims[t]
        im_inputs[t, 0:im.shape[0], 0:im.shape[1], :] = im
    return im_inputs

def prep_im_for_inputs(im, pixel_means, im_scale):
    """Mean subtract and scale an image for use in a inputs."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return im

def get_input_images(cfg, trajdb):
    """Get frame images from files."""
    n_steps = len(trajdb)
    im_list = []
    im_scales = []

    i = 0
    for t in range(n_steps):
        im = cv2.imread(trajdb[t]['image'])
        if trajdb[t]['flipped']:
            im = im[:, ::-1, :]
        im_scale = trajdb[t]['im_scale']

        # Scale the image
        im_shape = im.shape
        im = prep_im_for_inputs(im, cfg.PIXEL_MEANS, im_scale)

        im_list.append(im)
        im_scales.append(im_scale)

    # Create a numpy array to hold the input images
    im_inputs = im_list_to_inputs(im_list)

    return im_inputs, im_scales

def get_input_conv5s(cfg, trajdb):
    """Get frame conv5 features from files."""
    n_steps = len(trajdb)
    im_scales = []
    im_list = []

    # Read data
    conv5_path = os.path.join(data_root, 'conv5',
                                '%s.h5' % trajdb[0]['video_name'])
    with h5py.File(conv5_path, 'r') as hf:
        for t in range(n_steps):
            conv5_data = np.array(hf.get(str(trajdb[t]['fr'])))
            im_list.append(conv5_data)
            im_scales.append(trajdb[t]['im_scale'])

    # im_list to np array
    max_shape = np.array([im.shape for im in im_list]).max(axis=0)
    im_inputs = np.zeros((n_steps, max_shape[0], max_shape[1], max_shape[2]),
                        dtype=np.float32)
    for t in range(n_steps):
        im = im_list[t]
        im_inputs[t, 0:im.shape[0], 0:im.shape[1], :] = im

    return im_inputs, im_scales

def _sample_rois(cfg, trajdb, roidb, n_rois, n_pos_rois, n_neg_rois,
                random=True, insert_gt=False):
    """Sample RoIs from the box candidates."""
    assert n_pos_rois >= 1
    obj_id = trajdb['obj_id']
    i_obj_id = roidb['obj_ids'].index(obj_id)

    rank_inds = rank_nearest_boxes(trajdb['gt_box'], roidb['boxes'])
    nbr_inds = rank_inds[0:cfg.TRAIN.N_ROI_SAMPLES]
    boxes = roidb['boxes'][nbr_inds, :]
    overlaps = roidb['gt_overlaps'][nbr_inds, i_obj_id]
    max_overlaps = roidb['gt_overlaps'][nbr_inds, :].max(axis=1)

    th = cfg.TRAIN.POS_TH
    pos_inds = np.where(overlaps >= th)[0]
    neg_inds = np.where((overlaps < th) & (max_overlaps >= th))[0]
    bg_inds = np.where(overlaps < th)[0]

    # n_pos_rois = np.minimum(n_pos_rois, pos_inds.size)
    # n_neg_rois = np.minimum(n_neg_rois, neg_inds.size)
    # n_bg_rois = np.minimum(n_rois - n_pos_rois - n_neg_rois, bg_inds.size)

    n_pos_rois = n_pos_rois if pos_inds.size > 0 else 0
    n_neg_rois = n_neg_rois if neg_inds.size > 0 else 0
    n_bg_rois = n_rois - n_pos_rois - n_neg_rois

    if random:
        if n_pos_rois > 0:
            pos_inds = np.random.choice(pos_inds, size=n_pos_rois,
                                         replace=True)
        if n_neg_rois > 0:
            neg_inds = np.random.choice(neg_inds, size=n_neg_rois,
                                         replace=True)
        if n_bg_rois > 0:
            bg_inds = np.random.choice(bg_inds, size=n_bg_rois,
                                         replace=False)
    else:
        pos_inds = pos_inds[:n_pos_rois]
        neg_inds = neg_inds[:n_neg_rois]
        bg_inds = bg_inds[:n_bg_rois]

    if insert_gt:
        gt_inds = np.where(overlaps == 1.0)[0]
        if gt_inds.size > 0:
            pos_inds[-1] = gt_inds[0]

    keep_inds = np.concatenate([neg_inds, bg_inds, pos_inds])

    rois = boxes[keep_inds]
    overlaps = overlaps[keep_inds]
    return rois, overlaps

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

