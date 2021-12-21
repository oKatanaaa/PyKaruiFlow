import pykaruiflow as kf
import numpy as np


if __name__ == '__main__':
    t = kf.TensorHolder(shape=[1, 1], dtype='float32')
    feed_dict = {t: 3 * np.ones((1, 1), dtype='float32')}
    print(t.forward(feed_dict))
