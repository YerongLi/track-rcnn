import time
import os

import numpy as np
import tensorflow as tf
from net import TrackRCNN
from minibatch import get_minibatch

class SolverWrapper(object):
    """
    """

    def __init__(self, cfg, m, train_trajdb, train_roidb, output_dir='tmp'):
        self.m = m 
        self.train_trajdb = train_trajdb
        self.train_roidb = train_roidb
        self.output_dir = os.path.join('outputs', output_dir)

        self.lr = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads = tf.gradients(m.loss, tvars)
        grads, _ = tf.clip_by_global_norm(tf.gradients(m.loss, tvars),
                                            cfg.TRAIN.MAX_GRAD_NORM)
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif cfg.TRAIN.OPTIMIZER == 'sgdm':
            optimizer = tf.train.MomentumOptimizer(self.lr, cfg.TRAIN.MOMENTUM)
        elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                    decay=0.9, momentum=cfg.TRAIN.MOMENTUM, use_locking=False)
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        else:
            raise NotImplementedError

        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # For checkpoint
        self.saver = tf.train.Saver()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.ckpt_path = os.path.join(self.output_dir, 'model.ckpt')
        self.log_path = os.path.join(self.output_dir, 'log.txt')

        # For trajdb sampling
        self._init_trajdb_sampling()

    def _init_trajdb_sampling(self):
        n_traj = len(self.train_trajdb)
        self.traj_pointers = [0 for _ in range(n_traj)]

        avoid occluded starting pooint
        for i in range(n_traj):
            start = self.traj_pointers[i]
            while(self.train_trajdb[i][start]['visible'] == 0):
                start += 1
            self.traj_pointers[i] = start 

        self.traj_running = [True for _ in range(n_traj)]
        self.n_steps_to_go = [len(self.train_trajdb[i]) - self.traj_pointers[i]
                                for i in range(n_traj)]

    def _sample_trajdb(self, n_steps):
        # sample
        prob = np.array(self.n_steps_to_go) / sum(self.n_steps_to_go)
        ind = np.random.choice(len(self.train_trajdb), 1, p=prob)[0]
        start = self.traj_pointers[ind]
        end = min(start + n_steps, len(self.train_trajdb[ind]))

        # maintain
        self.traj_pointers[ind] = end
        self.n_steps_to_go[ind] =\
            len(self.train_trajdb[ind]) - self.traj_pointers[ind]
        if self.n_steps_to_go[ind] == 0:
            self.traj_running = False

        return self.train_trajdb[ind][start:end]

    def train_model(self, cfg, sess, max_iters):
        """Network traininig loop."""
        n_steps = cfg.TRAIN.N_STEPS
        m = self.m
        train_trajdb = self.train_trajdb
        train_roidb = self.train_roidb

        avg_loss = 0.0
        np.random.shuffle(perm)
        self.info("#iterations per epoch: %d" % iters_per_epoch)

        epoch = 0
        for iter in range(max_iters):
            start_time = time.time()

            # Set learning rate
            sess.run(tf.assign(self.lr, cfg.TRAIN.INITIAL_LR))

            # Prepare training batch
            trajdb = train_trajdb[idx]

            # Find corresponding roidb
            video_id = trajdb[0]['video_id']
            roidb = train_roidb[video_id]

            # Get batch data
            batch_data = get_minibatch(cfg, trajdb, roidb,
                                        dtype='conv5', random=True)
            predict_mask = np.ones(n_steps)
            predict_mask[0] = 0
            
            feed_dict = {m.im_inputs: batch_data['im_inputs'],
                        m.roi_inputs: batch_data['roi_inputs'],
                        m.score_targets: batch_data['score_targets'],
                        m.predict_mask: predict_mask,
                        m.keep_prob: 1.0}

            # Run training
            loss, score_preds, final_state, _ =\
            sess.run([m.loss, m.score_preds, m.final_state, self.train_op],
                        feed_dict=feed_dict)

            avg_loss += loss
            time_cost = time.time() - start_time

            self.info('iter: %d / %d, train_loss: %.4f, lr: %f, time: %.1f' %\
                    (iter+1, max_iters, loss, self.lr.eval(), time_cost))

            # Debug
            score_targets = feed_dict[m.score_targets]
            msg = '\n'
            for i in range(score_preds.shape[1]):
                row = ''
                for t in range(score_preds.shape[0]):
                    row += '(%.3f, %.3f)'%(score_targets[t,i], score_preds[t,i])
                msg += row + '\n'
            print(msg)

            # Finish an epoch
            if iter_in_epoch == iters_per_epoch - 1:
                avg_loss /= iters_per_epoch
                self.info('Epoch: %d , avg_loss: %.4f' %\
                            (epoch+1, avg_loss))

                self.saver.save(sess, self.ckpt_path)
                self.info("Save checkpoint.")

                np.random.shuffle(perm)
                train_err = 0.0
                epoch += 1
                avg_loss = 0

    def restore(self, cfg, sess, pretrain_path):
        """
        Restore variables from pretrained model.
        """
        if cfg.TRAIN.LOAD_PRETRAINED_CNN_ONLY:
            variables = tf.all_variables()
            name_list = ['conv1/weights',
                        'conv1/biases',
                        'conv2/weights',
                        'conv2/biases',
                        'conv3/weights',
                        'conv3/biases',
                        'conv4/weights',
                        'conv4/biases',
                        'conv5/weights',
                        'conv5/biases',
                        'fc6/weights',
                        'fc6/biases',
                        'fc7/weights',
                        'fc7/biases']

            pretrain_dict = {}

            for var in variables:
                var_name = var.name[:-2]
                if var_name in name_list:
                    print('Loading {} from checkpoint.'.format(var.name))
                    pretrain_dict[var_name] = var

            saver = tf.train.Saver(pretrain_dict)
            saver.restore(sess, pretrain_path)
        else:
            self.saver.restore(sess, pretrain_path)

    def info(self, string):
        """Log information."""
        print(string)
        logfile = open(self.log_path, "a")
        logfile.write(string+"\n")

def train_net(cfg, train_trajdb, train_roidb, output_dir, pretrain, use_rnn,
                max_iters):
    """ """
    with tf.Graph().as_default(), tf.Session() as sess:

        # Build 
        m = TrackRCNN(cfg)

        # Solve
        sw = SolverWrapper(cfg, m, train_trajdb, train_roidb, output_dir)

        # Initialize
        tf.initialize_all_variables().run()
        if pretrain == None:
            m.load_cnn_params(sess)
            print('Load the pretrained Caffe model.')
        else:
            sw.restore(cfg, sess, os.path.join('outputs',pretrain,'model.ckpt'))
            print('Load from the checkpoint.')

        # Train
        sw.train_model(cfg, sess, max_iters)


