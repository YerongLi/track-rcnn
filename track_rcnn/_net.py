import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops

from cnn import cnn_config
from ops import roi_pooling_op, roi_pooling_op_grad

class TrackRCNN(object):
    """
    """

    def __init__(self, cfg):
        # Configuration
        self._n_rois = n_rois = cfg.TRAIN.N_ROIS
        self._n_max_h = max_h = cfg.TRAIN.MAX_H
        self._n_max_w = max_w = cfg.TRAIN.MAX_W
        self._cnn_model = cnn_model = cnn_config[cfg.CNN_MODEL_NAME]
        self._spatial_scale = spatial_scale = cnn_model.spatial_scale
        self._pooled_h = pooled_h = cnn_model.pooled_h
        self._pooled_w = pooled_w = cnn_model.pooled_w
        self._rnn_size = rnn_size = cfg.RNN_SIZE

        # Inputs and targets
        self._im_inputs = tf.placeholder(tf.float32,
                        [None, None, None, None, 3], name='im_inputs')
        self._roi_inputs = tf.placeholder(tf.float32,
                        [None, n_rois, 5], name='roi_inputs')
        self._score_targets = tf.placeholder(tf.float32,
                        [None, n_rois], name='score_targets')
        self._keep_prob = tf.placeholder(tf.float32)

        # Tensor arrays
        roi_inputs_shape = array_ops.shape(self.roi_inputs)
        (n_steps, _, _) = array_ops.unpack(roi_inputs_shape, 3)

        inputs_got_shape = self.roi_inputs.get_shape().with_rank(3)
        (const_time_steps, const_batch_size, const_depth) =\
            inputs_got_shape.as_list()

        output_ta = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=n_steps,
            tensor_array_name='output_ta')

        # with tf.variable_scope('rnn') as scope:
        #     cell = rnn_cell.GRUCell(rnn_size)
        #     self._init_state = init_state =\
        #         cell.zero_state(n_rois, tf.float32)

        # Prepare for the recurrence
        t = array_ops.constant(0, dtype=tf.int32, name="time")
        # state = init_state
        self._init_state = state = tf.zeros([n_rois, 4096])

        def _choose_roi_index(score_t, visible_th):
            """
            Choose the RoI indices corresponding to the best score.
            """
            roi_index_t = tf.argmax(score_t, 0)
            max_score = tf.reduce_max(score_t)
            visible_t = tf.greater(max_score, visible_th)
            return roi_index_t, visible_t

        def _update_state(state_t, state_tm1, roi_index_t, visible_t):
            """
            Update the state according to the chosen RoI indices. 
            """
            chosen_state_t = tf.gather(state_t, roi_index_t)
            cond = tf.to_float(visible_t)
            new_state =\
                cond * tf.tile(tf.reshape(chosen_state_t, (1,-1)), (n_rois, 1))\
                + (1-cond) * state_tm1
            return new_state

        def _time_step(t, state_tm1, output_ta_t):
            """Take a time step."""
            ims_t = tf.gather(self.im_inputs, t)
            rois_t = tf.gather(self.roi_inputs, t)
            score_targets_t = tf.gather(self.score_targets, t)

            # Conv 1~5
            conv1_layers = cnn_model.conv1({'data': ims_t}, trainable=True)
            conv1_t = conv1_layers.get_output()

            conv5_layers = cnn_model.conv5({'data': conv1_t}, trainable=True)
            conv5_t = conv5_layers.get_output()

            # RoI Pooling
            roi_pool_t, _ = roi_pooling_op.roi_pool(conv5_t, rois_t,
                                            pooled_h, pooled_w, spatial_scale)

            # FC 6~7
            fc7_layers = cnn_model.fc7({'data': roi_pool_t}, keep_prob=None,
                                        trainable=False)
            fc7_t = fc7_layers.get_output() 

            # RNN
            # with tf.variable_scope('rnn') as scope:
            #     cell_output_t, state_t = cell(fc7_t, state_tm1)
            state_t = fc7_t

            # with tf.variable_scope('my_fc8') as scope:
            #     init_weights = tf.truncated_normal_initializer(0.0,stddev=0.001)
            #     init_biases = tf.constant_initializer(0.0)
            #     weights = tf.get_variable("weights", [rnn_size*2, rnn_size*2],
            #                                 initializer=init_weights)
            #     biases = tf.get_variable("biases", rnn_size*2,
            #                                 initializer=init_biases)
            #     # inputs = tf.concat(1, [state_tm1, state_t])
            #     inputs = state_t
            #     fc8_t = tf.nn.relu(tf.nn.xw_plus_b(inputs, weights, biases))
            #     fc8_t = tf.nn.dropout(fc8_t, self.keep_prob)
            #     # fc8_t = tf.square(state_t - state_tm1)

            # Predict scores
            with tf.variable_scope('my_fc8') as scope:
                init_weights = tf.truncated_normal_initializer(0.0,stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
                weights = tf.get_variable("weights", [4096, 1],
                                            initializer=init_weights)
                biases = tf.get_variable("biases", 1,
                                            initializer=init_biases)
                score_preds_t = tf.nn.xw_plus_b(state_t, weights, biases)
                score_preds_t = tf.maximum(tf.minimum(score_preds_t, 1.0), 0.0)

            output_ta_t = output_ta_t.write(t, score_preds_t)

            roi_index_t, visible_t =\
                _choose_roi_index(score_targets_t, cfg.TRAIN.POS_TH)
            new_state =\
                _update_state(state_t, state_tm1, roi_index_t, visible_t)

            return (t+1, new_state, output_ta_t)

        # Run recurence
        (final_t, final_state, final_output_ta) = control_flow_ops.while_loop(
            cond=lambda t, _1, _2: t < n_steps,
            body=_time_step,
            loop_vars=(t, state, output_ta))

        self._final_state = final_state
        self._score_preds = tf.reshape(final_output_ta.pack(), [-1, n_rois])

        # Loss
        self._loss = loss = tf.sqrt(tf.reduce_mean(
            tf.square(self.score_preds - self.score_targets)))
        # self._loss = loss =\
        #     tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     self.score_preds[1:, :], self.score_targets[1:, :]))

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
    def n_rois(self):
        return self._n_rois

    @ property
    def n_max_h(self):
        return self._n_max_h

    @ property
    def n_max_w(self):
        return self._n_max_w

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

