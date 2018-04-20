import numpy as np
import time

from minibatch import get_input_images, get_input_conv5s
from datasets.videodb import rank_nearest_boxes
from motion_prior import ConstVelocityMP

def track_obj(cfg, sess, m, trajdb, roidb, n_rois):
    """ """
    n_steps = len(trajdb)
    state =  m.init_state.eval()

    # Initialize motion prior
    mp = ConstVelocityMP(n_steps)
    motion_history = []
    boxes = []
    confs = []
    pred_box = trajdb[0]['gt_box']

    rois = []
    score_preds = []
    avg_loss = 0.0
    avg_time = 0.0
    for t in range(n_steps):
        # Prepare data
        fr = trajdb[t]['fr']

        input_data = get_frame(cfg, trajdb[t], roidb[fr], pred_box, n_rois,
                                'conv5')

        # Deal with initialization
        predict_mask = np.ones(1)
        if t == 0:
            idx = input_data['score_targets'].reshape([-1]).argmax()
            input_data['roi_inputs'][:, :] = input_data['roi_inputs'][:, idx]
            input_data['score_targets'][:,:] =input_data['score_targets'][:,idx]
            predict_mask[0] = 0

        # Make feed dict
        feed_dict = {m.im_inputs: input_data['im_inputs'],
                    m.roi_inputs: input_data['roi_inputs'],
                    # m.score_targets: input_data['score_targets'],
                    m.init_state: state,
                    m.predict_mask: predict_mask,
                    m.keep_prob: 1.0}
        im_scale = input_data['im_scales'][0]

        # Run testing
        start_time = time.time()
        score_preds_t, state = sess.run([m.score_preds, m.final_state],
                                        feed_dict=feed_dict)
        loss_t = np.sqrt(np.mean(np.square(
                    input_data['score_targets'] - score_preds_t)))
        time_cost = time.time() - start_time

        # Choose the box
        ind = np.argmax(score_preds_t, axis=-1)
        if score_preds_t[:, ind] < cfg.TRAIN.POS_TH:
            box = pred_box
        else:
            chosen_roi = input_data['roi_inputs'][0, ind, 1:].reshape(4)
            box = chosen_roi / im_scale
        boxes.append(box)
        confs.append(score_preds_t[0, ind])

        # Update results
        avg_loss += loss_t
        avg_time += time_cost
        rois.append(input_data['roi_inputs'][:, :, 1:] / im_scale)
        score_preds.append(score_preds_t)
        pred_box, motion_history = mp(box, motion_history)
        print(box, pred_box)
        print('step: %d, loss: %.4f, time: %.2f' % (t, loss_t, time_cost))

    avg_loss /= n_steps
    avg_time /= n_steps
    print('n_steps: %d, avg_loss: %.4f, avg_time: %.2f'\
            % (n_steps, avg_loss, avg_time))
    return boxes, confs, rois, score_preds

def get_frame(cfg, trajdb, roidb, cbox, n_rois, dtype='image'):
    """Given a trajdb(trajectory batch), construct a minibatch sample."""
    n_rois = cfg.TRAIN.N_ROIS
    n_pos_rois = np.round(cfg.TRAIN.POS_FRACTION * n_rois).astype('int32')
    n_neg_rois = np.round(cfg.TRAIN.NEG_FRACTION * n_rois).astype('int32')

    # Get the input images, preprcess
    if dtype == 'image':
        im_inputs, im_scales = get_input_images(cfg, [trajdb])
    elif dtype == 'conv5':
        im_inputs, im_scales = get_input_conv5s(cfg, [trajdb])

    # Get the input RoI and ground truth scores
    im_rois, overlaps = _get_roi_candidates(cfg, trajdb, roidb, cbox, n_rois)
    n_rois = im_rois.shape[0]

    roi_inputs = np.zeros((1, n_rois, 5), dtype=np.float32)
    score_targets = np.zeros((1, n_rois), dtype=np.float32)

    # Add to RoIs inputs
    rois = _project_im_rois(im_rois, im_scales[0])
    batch_ind = np.zeros((rois.shape[0], 1))
    rois_this_image = np.hstack((batch_ind, rois))
    roi_inputs[0, :, :] = rois_this_image

    # Add to targets
    scores = overlaps
    # scores = (overlaps >= cfg.TRAIN.POS_TH)
    score_targets[0, :] = scores.astype('float32')

    batch_data = {'im_inputs': im_inputs, 
                'im_scales': im_scales,
                'roi_inputs': roi_inputs,
                'score_targets': score_targets}
    return batch_data

def _get_roi_candidates(cfg, trajdb, roidb, cbox, n_rois):
    """Sample RoIs from the box candidates."""
    obj_id = trajdb['obj_id']
    i_obj_id = roidb['obj_ids'].index(obj_id)
    boxes = roidb['boxes']
    overlaps = roidb['gt_overlaps'][:, i_obj_id]

    rank_inds = rank_nearest_boxes(cbox, boxes)
    nbr_inds = rank_inds[0:n_rois]

    rois = boxes[nbr_inds]
    overlaps = overlaps[nbr_inds]
    return rois, overlaps

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

