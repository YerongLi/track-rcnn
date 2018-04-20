#
# Depricated. Use roi_pooling_ops instead
# 
import numpy as np
import tensorflow as tf

def roi_pool(input_data, input_rois,
                spatial_scale, pooled_h, pooled_w,
                name='roi_pool', debug=False):
    """
    Create an RoI Polling layer in Tensorflow.

    Arguments:
    ----------
        input_data: n_data X height_ X width_ X channel
                    (e.g. 2 X 120 X 67 X 512)
        input_rois: n_rois X 5
                    (e.g. 128 X 5)
        spatial_scale: float
            Defined as in the RoI pooling layer.

    Returns:
    --------
        output: n_rois X pooled_h X pooled_w X channel
                (e.g. 128 X 7 X 7 X 512)
    """
    with tf.variable_scope(name) as scope:
        # 
        # Compute the grid of bins.
        #

        # Get the input shape (Need to speicify n_rois for iteration)
        input_shape = input_data.get_shape()
        n_rois = int(input_rois.get_shape()[0])
        height_ = int(input_shape[1])
        width_ = int(input_shape[2])
        channel = int(input_shape[3])

        # rois_batch_ind: h_rois X 1
        roi_batch_ind = input_rois[:, 0]
        # rois: h_rois X 4
        rois = tf.round(input_rois[:, 1:] * spatial_scale)
        roi_start_w = tf.minimum(tf.maximum(rois[:, 0], 0.), width_-1)
        roi_start_h = tf.minimum(tf.maximum(rois[:, 1], 0.), height_-1)
        roi_end_w = tf.minimum(tf.maximum(rois[:, 2], 0.), width_)
        roi_end_h = tf.minimum(tf.maximum(rois[:, 3], 0.), height_)

        # roi_h, roi_w: h_rois X 1
        roi_h = tf.maximum(roi_end_h - roi_start_h, 1.)
        roi_w = tf.maximum(roi_end_w - roi_start_w, 1.)
        # bin_size_w, bin_size_h: h_rois X 1
        bin_size_w = roi_w / pooled_w
        bin_size_h = roi_h / pooled_h
        _bin_size_w = tf.to_int32(tf.maximum(1., bin_size_w))
        _bin_size_h = tf.to_int32(tf.maximum(1., bin_size_h))

        # hstart, hend: n_rois X pooled_h
        ph = tf.to_float(tf.range(0, pooled_h))
        hstart = tf.floor(\
            tf.tile(tf.reshape(bin_size_h, [n_rois,1]), [1,pooled_h])*\
            tf.tile(tf.reshape(ph, [1, pooled_h]), [n_rois, 1]))
        hend = tf.ceil(\
            tf.tile(tf.reshape(bin_size_h, [n_rois,1]), [1,pooled_h])*\
            tf.tile(tf.reshape(ph + 1, [1, pooled_h]), [n_rois, 1]))
        _roi_start_h =\
            tf.tile(tf.reshape(roi_start_h, [n_rois,1]), [1,pooled_h])
        # hstart = tf.minimum(tf.maximum(hstart + _roi_start_h, 0), height_)
        # hend = tf.minimum(tf.maximum(hend + _roi_start_h, 0), height_)

        # wstart, wend: n_rois X pooled_w
        pw = tf.to_float(tf.range(0, pooled_w))
        wstart = tf.floor(\
            tf.tile(tf.reshape(bin_size_w, [n_rois,1]), [1,pooled_w])*\
            tf.tile(tf.reshape(pw, [1, pooled_w]), [n_rois, 1]))
        wend = tf.ceil(\
            tf.tile(tf.reshape(bin_size_w, [n_rois,1]), [1,pooled_w])*\
            tf.tile(tf.reshape(pw + 1, [1, pooled_w]), [n_rois, 1]))
        _roi_start_w = \
            tf.tile(tf.reshape(roi_start_w, [n_rois,1]), [1,pooled_w])
        # wstart = tf.minimum(tf.maximum(wstart + _roi_start_w, 0), width_)
        # wend = tf.minimum(tf.maximum(wend + _roi_start_w, 0), width_)

        # Compute the flatten index of each spatial location
        # (A detour without using tf.gather_nd() in later Tensorflow verions):
        # index = h * width_ + w
        #       = (hstart + hi) * width_ + (wstart + wi)
        #       = (hstart * width_ + hstart) + (hi * width_ + wi)
        #       = index_start + index_iter

        # 
        # Prepare for indexing in each bin
        #

        # data_for_roi = tf.gather(input_data, tf.to_int32(roi_batch_ind))
        data_for_roi = tf.gather(input_data, tf.to_int32(roi_batch_ind))
        # flat_data: n_data X (height_ X width_) X channel
        flat_data = tf.reshape(data_for_roi, [n_rois, height_*width_, channel])

        # index_start: n_rois X pooled_h X pooeld_w
        index_start = tf.tile(tf.expand_dims(hstart * width_, dim=2),
                                [1, 1, pooled_w]) +\
                        tf.tile(tf.expand_dims(wstart, dim=1), [1, pooled_h, 1])

        # 
        # Index the data of each RoI, do the max pooling.
        #

        pooled = []
        for i in range(n_rois):
            h = tf.to_float(tf.range(0, _bin_size_h[i]))
            w = tf.to_float(tf.range(0, _bin_size_w[i]))

            # index_iter: _bin_size_h[i] X _bin_size_w[i]
            index_iter = tf.tile(tf.expand_dims(h * width_, dim=1),
                                tf.pack([1, _bin_size_w[i]])) +\
                        tf.tile(tf.expand_dims(w, dim=0),
                                tf.pack([_bin_size_h[i], 1]))

            # index: pooled_h X pooled_w X _bin_size_h[i] X _bin_size_w[i]
            index = tf.to_int32(
                tf.tile(tf.expand_dims(tf.expand_dims(index_start[i,:,:],2),3),
                        tf.pack([1, 1, _bin_size_h[i], _bin_size_w[i]])) +\
                tf.tile(tf.expand_dims(tf.expand_dims(index_iter, 0), 1),
                        [pooled_h, pooled_w, 1, 1]))

            # to_pool: pooled_h X pooled_w X _bin_size_h[i] X _bin_size_w[i]
            to_pool = tf.gather(flat_data[i, :, :], index)
            pooled.append(tf.expand_dims(tf.reduce_max(to_pool, [2, 3]), dim=0))

        output = tf.concat(0, pooled)

        # Debug, plot the grid of bins
        if debug:
            # _hstart, _hend, _wstart, _wend: n_rois X pooled_h X pooled_w
            _hstart = tf.tile(tf.expand_dims(hstart, 2), [1, 1, pooled_w])
            _hend = tf.tile(tf.expand_dims(hend, 2), [1, 1, pooled_w])
            _wstart = tf.tile(tf.expand_dims(wstart, 1), [1, pooled_h, 1])
            _wend = tf.tile(tf.expand_dims(wend, 1), [1, pooled_h, 1])
            # bins: n_rois X pooled_h X pooled_w X 4
            bins = tf.concat(3, 
                            [tf.expand_dims(_wstart, 3),
                            tf.expand_dims(_hstart, 3),
                            tf.expand_dims(_wend, 3),
                            tf.expand_dims(_hend, 3)])
            return output, bins

        return output


