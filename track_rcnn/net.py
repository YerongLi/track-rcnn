import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops

from cnn import cnn_config
from ops import roi_pooling_op, roi_pooling_op_grad

EPS = 1e-8

class TrackRCNN(object):
    """
    """

    def __init__(self, cfg, mode='train'):
        # Configuration
        self._cnn_model = cnn_model = cnn_config[cfg.CNN_MODEL_NAME]
        self._spatial_scale = spatial_scale = cnn_model.spatial_scale
        self._pooled_h = pooled_h = cnn_model.pooled_h
        self._pooled_w = pooled_w = cnn_model.pooled_w
        self._rnn_size = rnn_size = cfg.RNN_SIZE

        # Inputs and targets
        self._im_inputs = im_inputs = tf.placeholder(tf.float32,
            [None, None, None, cnn_model.conv5_size], name='im_inputs')
        self._roi_inputs = roi_inputs = tf.placeholder(tf.float32,
            [None, None, 5], name='roi_inputs')
        self._score_targets = score_targets = tf.placeholder(tf.float32,
            [None, None], name='score_targets')
        self._predict_mask = predict_mask =\
            tf.placeholder(tf.float32, [None], name='predict_mask')
        self._keep_prob = keep_prob =\
            tf.placeholder(tf.float32, name='keep_prob')

        # Get input shapes
        roi_inputs_shape = array_ops.shape(self.roi_inputs)
        (n_steps, n_rois, _) = array_ops.unpack(roi_inputs_shape, 3)

        inputs_got_shape = self.roi_inputs.get_shape().with_rank(3)
        (const_n_steps, const_n_rois, _) =\
            inputs_got_shape.as_list()

        # Prepare CNN inputs
        ims = im_inputs

        _rois = tf.reshape(self._roi_inputs[:, :, 1:], [-1, 4])
        batch_ind = tf.to_float(
                        tf.reshape(
                            tf.tile(
                                tf.reshape(
                                    tf.range(0, n_steps),
                                [-1, 1]),
                            tf.pack([1, n_rois])),
                        [-1, 1]))
        rois = tf.concat(1, [batch_ind, _rois])

        # # Conv 1~5
        # conv1_layers = cnn_model.conv1({'data': ims}, trainable=False)
        # conv1 = conv1_layers.get_output()

        # conv5_layers = cnn_model.conv5({'data': conv1}, trainable=False)
        # conv5 = conv5_layers.get_output()
        conv5 = im_inputs

        # RoI Pooling
        roi_pool, _ = roi_pooling_op.roi_pool(conv5, rois,
                                        pooled_h, pooled_w, spatial_scale)

        # FC 6~7
        fc6_layers = cnn_model.fc6({'data':roi_pool}, keep_prob,trainable=False)
        fc6 = fc6_layers.get_output() 

        # Prepare RNN inputs
        feat_size = int(fc6.get_shape()[-1])
        feats = tf.reshape(fc6, tf.pack([-1, n_rois, feat_size]))

        # RNN
        with tf.variable_scope('rnn') as scope:
            cell = rnn_cell.GRUCell(rnn_size)
            self._init_state = init_state =\
                cell.zero_state(1, tf.float32)

        def _choose_roi_index(score_t, visible_th):
            """
            Choose the RoI indices corresponding to the best score.
            """
            roi_index_t = tf.argmax(score_t, 0)
            max_score = tf.reduce_max(score_t)
            visible_t = tf.greater(max_score, visible_th)
            return roi_index_t, visible_t

        def _update_state(state_t, state_tm1, roi_index_t,
                            visible_t, predict_mask_t):
            """
            Update the state according to the chosen RoI indices. 
            """
            is_feed = tf.equal(predict_mask_t, 0)
            cond = tf.reshape(tf.logical_or(visible_t, is_feed), [1])
            chosen_state_t = tf.expand_dims(tf.gather(state_t, roi_index_t), 0)
            new_state = tf.select(cond, chosen_state_t, state_tm1)
            return new_state

        # Tensor arrays
        output_ta = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=n_steps,
            tensor_array_name='output_ta')

        feats_ta = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=n_steps,
            tensor_array_name='feats_ta')
        feats_ta = feats_ta.unpack(feats)

        score_targets_ta = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=n_steps,
            tensor_array_name='score_targets_ta')
        score_targets_ta = score_targets_ta.unpack(score_targets)

        predict_mask_ta = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=n_steps,
            tensor_array_name='predict_mask_ta')
        predict_mask_ta = predict_mask_ta.unpack(predict_mask)

        t = array_ops.constant(0, dtype=tf.int32, name="time")

        state = init_state

        def _time_step(t, state_tm1, output_ta_t):
            """Take a time step."""
            feats_t = feats_ta.read(t)
            score_targets_t = score_targets_ta.read(t)
            predict_mask_t = predict_mask_ta.read(t)

            with tf.variable_scope('rnn') as scope:
                states_tm1 = tf.tile(state_tm1, tf.pack([n_rois, 1]))
                cell_output_t, state_t =\
                    cell(tf.reshape(feats_t, [-1, feat_size]),
                        tf.reshape(states_tm1, [-1, rnn_size]))

            with tf.variable_scope('predict_score') as scope:
                init_weights =\
                    tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                weights = tf.get_variable("weights", [rnn_size, 1],
                                            initializer=init_weights)
                init_biases = tf.constant_initializer(0.0)
                biases = tf.get_variable("biases", 1, initializer=init_biases)
                score_uncapped = tf.nn.xw_plus_b(cell_output_t, weights, biases)
                score_preds_t = tf.maximum(tf.minimum(score_uncapped, 1.0), 0.0)

            output_ta_t = output_ta_t.write(t, score_preds_t)

            if mode == 'train' and cfg.TRAIN.CHOOSE_BY_SCORE_TARGETS:
                score_t = score_targets_t
            else:
                score_t = tf.reshape(score_preds_t, [-1])

            roi_index_t, visible_t =\
                _choose_roi_index(score_t, 0.5) # TODO
            new_state = _update_state(state_t, state_tm1, roi_index_t,
                                        visible_t, predict_mask_t)

            return (t+1, new_state, output_ta_t)

        # Run recurence
        (final_t, final_state, final_output_ta) = control_flow_ops.while_loop(
            cond=lambda t, _1, _2: t < n_steps,
            body=_time_step,
            loop_vars=(t, state, output_ta))

        self._final_state = final_state
        self._score_preds = score_preds = tf.reshape(final_output_ta.pack(),
                                                    tf.pack([n_steps, n_rois]))

        # Loss
        self._loss = loss = tf.sqrt(tf.div(tf.reduce_sum(
            tf.reduce_mean(tf.square(score_preds - score_targets), 1)\
            * predict_mask), tf.reduce_sum(predict_mask) + EPS))

    def load_cnn_params(self, sess, ignore_missing=True):
        data_path = self.cnn_model.params_path
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

    @ property
    def cnn_model(self):
        return self._cnn_model

    @ property
    def spatial_scale(self):
        return self._spatial_scale

    @ property
    def pooled_h(self):
        return self._pooled_h

    @ property
    def pooled_w(self):
        return self._pooled_w

    @ property
    def im_inputs(self):
        return self._im_inputs

    @ property
    def roi_inputs(self):
        return self._roi_inputs

    @ property
    def score_targets(self):
        return self._score_targets

    @ property
    def predict_mask(self):
        return self._predict_mask

    @ property
    def keep_prob(self):
        return self._keep_prob

    @ property
    def init_state(self):
        return self._init_state

    @ property
    def final_state(self):
        return self._final_state

    @ property
    def score_preds(self):
        return self._score_preds

    @ property
    def loss(self):
        return self._loss

