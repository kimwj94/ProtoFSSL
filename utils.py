import numpy as np
import tensorflow as tf

#Compute euclidian distance between two tensors
def calc_euclidian_dists(x, y, root=False):
    """
    Calculate euclidian distance between two 3D tensors.
    Args:
        x (tf.Tensor): embedding vector
        y (tf.Tensor): prototype
    Returns (tf.Tensor): 2-dim tensor with distances.
    """
    n = x.shape[0] # data 개수
    m = y.shape[0] # ptototype 개수
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1]) # embedding vector 가 10번 반복됨.
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1]) # 모든 prototype이 있음
    #return tf.math.pow(tf.reduce_sum(tf.math.pow(x - y, 2), 2), 0.5)
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)


# compute prototypes for each class
def get_prototype(z, s_label, num_class, add_noise=False, stddev = 0.0):
    z_prototypes = tf.reshape(z[:num_class*s_label], [num_class, s_label, z.shape[-1]])
    z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
    
    if add_noise:
        noise_layer = tf.keras.layers.GaussianNoise(stddev)
        z_prototypes = noise_layer(z_prototypes, training=True)

    return z_prototypes


def difference_model_norm_2_square(global_model, local_model):
    """Calculates the squared l2 norm of a model difference (i.e.
    local_model - global_model)
    Args:
        global_model: the model broadcast by the server
        local_model: the current, in-training model

    Returns: the squared norm

    """
    model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                           local_model,
                                           global_model)
    squared_norm = tf.square(tf.linalg.global_norm(model_difference))
    return squared_norm