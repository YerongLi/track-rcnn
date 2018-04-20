#! /bin/bash
convert=./caffe-tensorflow/convert.py

$convert data/caffe_models/alexnet/deploy.prototxt data/caffe_models/alexnet/net.caffemodel data/tensorflow_models/alexnet.npy data/tensorflow_models/alexnet.py
$convert data/caffe_models/caffenet/deploy.prototxt data/caffe_models/caffenet/net.caffemodel data/tensorflow_models/caffenet.npy data/tensorflow_models/caffenet.py
$convert data/caffe_models/vgg16/deploy.prototxt data/caffe_models/vgg16/net.caffemodel data/tensorflow_models/vgg16.npy data/tensorflow_models/vgg16.py
