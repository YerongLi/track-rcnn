import numpy as np

def _x1y1x2y2_to_cxcywh(x1y1x2y2):
    cx = (x1y1x2y2[0] + x1y1x2y2[2]) / 2.
    cy = (x1y1x2y2[1] + x1y1x2y2[3]) / 2.
    w = x1y1x2y2[2] - x1y1x2y2[0]
    h = x1y1x2y2[3] - x1y1x2y2[1]
    return np.array([cx, cy, w, h])

def _cxcywh_to_x1y1x2y2(cxcywh):
    x1 = cxcywh[0] - cxcywh[2] / 2.
    y1 = cxcywh[1] - cxcywh[3] / 2.
    x2 = cxcywh[0] + cxcywh[2] / 2.
    y2 = cxcywh[1] + cxcywh[3] / 2.
    return np.array([x1, y1, x2, y2])

class MotionPrior(object):

    def __init__(self):
        return

    def __call__(self):
        return


class ConstVelocityMP(MotionPrior):
    """
    Constant velocity motion prior
    """

    def __init__(self, n_steps):
        self.n_steps = n_steps

    def __call__(self, x_t, history):
        if len(history) >= self.n_steps:
            new_history = list(history[1:])
        else:
            new_history = list(history)
        _x_t = _x1y1x2y2_to_cxcywh(x_t)
        new_history.append(_x_t)

        if len(new_history) == 1:
            vel = 0
        else:
            vel = (new_history[-1] - new_history[0]) / (len(new_history) - 1)

        _pred = _x_t + vel
        pred = _cxcywh_to_x1y1x2y2(_pred)
        return pred, new_history

if __name__ == '__main__':
    # Debug

    n_steps = 10
    mp = ConstVelocityMP(n_steps)

    X = np.array([
        [1, 1, 1, 1],
        [2, 1, 1.1, 2],
        [3, 1, 0.9, 4],
        [4, 1, 1.2, 8],
        [5, 1, 1.1, 16],
        [6, 1, 0.95, 32]
        ])

    # X = np.array([
    #     [1, 1, 1, 1]
    #     ])

    pred = np.array([0, 0, 0, 0])
    history = []
    for t in range(100):
        if t < X.shape[0]:
            inputs = X[t]
        else:
            inputs = pred
        pred, history = mp(inputs, history)
        # print(inputs)

