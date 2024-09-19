import pykaruiflow as kf
import numpy as np


if __name__ == '__main__':
    t = kf.tensor(3 * np.ones((1, 1), dtype='float32'))
    print(t)
