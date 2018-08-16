"""
Resnet version 2, "preactivation" variant

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
"""
from save_kernels import save_kernels
import sys
import os
import numpy as np
import tensorflow as tf
import deep_ckern as dkern
import deep_ckern.resnet
import pickle_utils as pu

def main(_):
    FLAGS = tf.app.flags.FLAGS
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    DEPTH = FLAGS.depth

    path = FLAGS.path
    def file_for(name, fmt):
        return os.path.join(path, "{}_{:02d}.{}".format(name, FLAGS.seed, fmt))
    params_file = file_for('params', 'pkl.gz')
    gram_file = file_for('gram', 'npy')
    Kxvx_file = file_for('Kxvx', 'npy')
    Kxtx_file = file_for('Kxtx', 'npy')
    Kv_diag_file = file_for('Kv_diag', 'npy')
    Kt_diag_file = file_for('Kt_diag', 'npy')

    if DEPTH % 6 != 2:
        raise ValueError('DEPTH must be 6n + 2:', DEPTH)

    block_depth = (DEPTH - 2) // 6

    params = {'depth': DEPTH}
    print("Params:", params)
    pu.dump(params, params_file)
    with tf.device("cpu:0"):
        kern = dkern.resnet.ResnetKernel(
            input_shape=[1, 28, 28],
            block_sizes=[block_depth]*FLAGS.n_blocks,
            block_strides=[1, 2, 2, 2, 2, 2, 2][:FLAGS.n_blocks],
            kernel_size=3,
            conv_stride=1,
            recurse_kern=dkern.ExReLU(),
            var_weight=1., #scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)**2,
            var_bias=0.,
            data_format='NCHW',
        )
    save_kernels(kern, FLAGS.n_gpus, gram_file, Kxvx_file, Kxtx_file,
                 n_max=FLAGS.n_max, Kv_diag_file=Kv_diag_file,
                 Kt_diag_file=Kt_diag_file)


if __name__ == '__main__':
    f = tf.app.flags
    f.DEFINE_integer('n_gpus', 1, 'Number of GPUs to use')
    f.DEFINE_integer('n_blocks', 3, 'Number of blocks to use')
    f.DEFINE_integer('seed', None, 'seed')
    f.DEFINE_integer('depth', 32, 'depth of the resnet')
    f.DEFINE_integer('n_max', 400, 'max number of examples to simultaneously compute the kernel of')
    f.DEFINE_string('path', "/scratch/ag919/grams_resnet/", 'save path')
    tf.app.run()
