import tensorflow as tf
import numpy as np

oldl = np.array([  [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])


newl = tf.slice(oldl, [0, 0], [-1, 1])
print('{}'.format(newl))
