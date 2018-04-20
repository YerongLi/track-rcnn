import os
from caffe_tensorflow.network import Network
from datasets.add_path import data_root

#
# CaffeNet
# 

class CaffeNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(1000, relu=False, name='fc8')
             .softmax(name='prob'))

class CaffeNetConv1(Network):
    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1'))

class CaffeNetConv5(Network):
    def setup(self):
        (self.feed('data')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5'))

class CaffeNetFC6(Network):
    def setup(self):
        (self.feed('data')
             .fc(4096, name='fc6')
             .dropout(self.keep_prob, name='drop6'))

class CaffeNetFC7(Network):
    def setup(self):
        (self.feed('data')
             .fc(4096, name='fc7')
             .dropout(self.keep_prob, name='drop7'))

class CNNConfig(object):
    def __init__(self, conv1, conv5, fc6, fc7, params_path,
                conv5_size,
                spatial_scale, pooled_h, pooled_w):
        self.conv1  = conv1
        self.conv5  = conv5
        self.fc6  = fc6
        self.fc7  = fc7
        self.params_path = params_path

        self.conv5_size = conv5_size

        self.spatial_scale = spatial_scale
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w

cnn_config = {
        'caffenet': CNNConfig(CaffeNetConv1,
                                CaffeNetConv5,
                                CaffeNetFC6,
                                CaffeNetFC7,
                                os.path.join(data_root,
                                    'tensorflow_models/caffenet.npy'),
                                256,
                                1/16., 6, 6)}

