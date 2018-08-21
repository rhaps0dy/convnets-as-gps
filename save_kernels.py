"""
Compute deep kernels with a bunch of different hyperparameters and save them to
disk
"""
import numpy as np
import tensorflow as tf
from gpflow import settings
import deep_ckern as dkern
import tqdm
import pickle_utils as pu
import sys
import os
import gpflow


def mnist_1hot_all():
    from tensorflow.examples.tutorials.mnist import input_data
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    d = input_data.read_data_sets("MNIST_data/", one_hot=True)
    r = tuple(a.astype(settings.float_type) for a in [
        d.train.images, d.train.labels,
        d.validation.images, d.validation.labels,
        d.test.images, d.test.labels])
    tf.logging.set_verbosity(old_v)
    return r


def create_kern(ps):
    if ps['seed'] == 1234:
        return dkern.DeepKernel(
            [1, 28, 28],
            filter_sizes=[[5, 5], [2, 2], [5, 5], [2, 2]],
            recurse_kern=dkern.ExReLU(multiply_by_sqrt2=True),
            var_weight=1.,
            var_bias=1.,
            padding=["VALID", "SAME", "VALID", "SAME"],
            strides=[[1, 1]] * 4,
            data_format="NCHW",
            skip_freq=-1,
        )

    if 'skip_freq' not in ps:
        ps['skip_freq'] = -1
    if ps['nlin'] == 'ExReLU':
        recurse_kern = dkern.ExReLU(multiply_by_sqrt2=True)
    else:
        recurse_kern = dkern.ExErf()
    return dkern.DeepKernel(
        [1, 28, 28],
        filter_sizes=[[ps['filter_sizes'], ps['filter_sizes']]] * ps['n_layers'],
        recurse_kern=recurse_kern,
        var_weight=ps['var_weight'],
        var_bias=ps['var_bias'],
        padding=ps['padding'],
        strides=[[ps['strides'], ps['strides']]] * ps['n_layers'],
        data_format="NCHW",
        skip_freq=ps['skip_freq'],
    )


def compute_big_Kdiag(sess, kern, n_max, X, n_gpus=1):
    """
    Compute Kdiag for a "big" data set `X`.
    `X` fits in memory, but the tensors required to compute the whole diagonal
    of the kernel matrix of `X` don't fit in GPU memory.
    """
    N = X.shape[0]
    slices = list(slice(j, j+n_max) for j in range(0, N, n_max))
    K_ops = []
    for i in range(n_gpus):
        with tf.device("gpu:{}".format(i)):
            X_ph = tf.placeholder(settings.float_type, [None, X.shape[1]], "X_ph")
            Kdiag = kern.Kdiag(X_ph)
            K_ops.append((X_ph, Kdiag))

    out = np.zeros([N], dtype=settings.float_type)
    for j in tqdm.trange(0, len(slices), n_gpus):
        feed_dict = {}
        ops = []
        for (X_ph, Kdiag), j_s in zip(K_ops, slices[j:j+n_gpus]):
            feed_dict[X_ph] = X[j_s]
            ops.append(Kdiag)
        results = sess.run(ops, feed_dict=feed_dict)

        for r, j_s in zip(results, slices[j:j+n_gpus]):
            out[j_s] = r
    return out


def compute_big_K(sess, kern, n_max, X, X2=None, n_gpus=1):
    """
    Compute the kernel matrix between `X` and `X2`.
    """
    N = X.shape[0]
    N2 = N if X2 is None else X2.shape[0]

    # Make a list of all the point kernel matrices to be computed
    if X2 is None or X2 is X:
        diag_symm = True
        slices = list((slice(j, j+n_max), slice(i, i+n_max))
                      for j in range(0, N, n_max)
                      for i in range(j, N2, n_max))
    else:
        diag_symm = False
        slices = list((slice(j, j+n_max), slice(i, i+n_max))
                      for j in range(0, N, n_max)
                      for i in range(0, N2, n_max))

    # Make the required kernel ops and placeholders for each GPU
    K_ops = []
    for i in range(n_gpus):
        with tf.device("gpu:{}".format(i)):
            X_ph = tf.placeholder(settings.float_type, [None, X.shape[1]], "X_ph")
            X2_ph = tf.placeholder(settings.float_type, X_ph.shape, "X2_ph")
            K_cross = kern.K(X_ph, X2_ph)
            if diag_symm:
                K_symm = kern.K(X_ph, None)
            else:
                K_symm = None
            K_ops.append((X_ph, X2_ph, K_cross, K_symm))

    # Execute on all GPUs concurrently
    out = np.zeros((N, N2), dtype=settings.float_type)
    for j in tqdm.trange(0, len(slices), n_gpus):
        feed_dict = {}
        ops = []
        for (X_ph, X2_ph, K_cross, K_symm), (j_s, i_s) in (
                zip(K_ops, slices[j:j+n_gpus])):
            if j_s == i_s and diag_symm:
                feed_dict[X_ph] = X[j_s]
                ops.append(K_symm)
            else:
                feed_dict[X_ph] = X[j_s]
                if X2 is None:
                    feed_dict[X2_ph] = X[i_s]
                else:
                    feed_dict[X2_ph] = X2[i_s]
                ops.append(K_cross)
        results = sess.run(ops, feed_dict=feed_dict)

        for r, (j_s, i_s) in zip(results, slices[j:j+n_gpus]):
            out[j_s, i_s] = r
            if j_s != i_s and diag_symm:
                out[i_s, j_s] = r.T
    return out


def save_kernels(kern, N_train, N_vali, n_gpus, gram_file, Kxvx_file,
                 Kxtx_file, n_max=400, Kv_diag_file=None, Kt_diag_file=None):
    X, _, Xv, _, Xt, _ = mnist_1hot_all()
    Xv = np.concatenate([X[N_train:, :], Xv], axis=0)[:N_vali, :]
    X = X[:N_train]

    sess = gpflow.get_default_session()
    if os.path.isfile(gram_file):
        print("Skipping Kxx")
    else:
        print("Computing Kxx")
        Kxx = compute_big_K(sess, kern, n_max=n_max, X=X, n_gpus=n_gpus)
        np.save(gram_file, Kxx, allow_pickle=False)

    if os.path.isfile(Kxvx_file):
        print("Skipping Kxvx")
    else:
        print("Computing Kxvx")
        Kxvx = compute_big_K(sess, kern, n_max=n_max, X=Xv, X2=X, n_gpus=n_gpus)
        np.save(Kxvx_file, Kxvx, allow_pickle=False)

    if os.path.isfile(Kxtx_file):
        print("Skipping Kxtx")
    else:
        print("Computing Kxtx")
        Kxtx = compute_big_K(sess, kern, n_max=n_max, X=Xt, X2=X, n_gpus=n_gpus)
        np.save(Kxtx_file, Kxtx, allow_pickle=False)

    if Kv_diag_file is not None:
        if os.path.isfile(Kv_diag_file):
            print("Skipping Kv_diag")
        else:
            print("Computing Kv_diag")
            Kv_diag = compute_big_Kdiag(sess, kern, n_max=n_max*n_max,
                                        X=Xv, n_gpus=n_gpus)
            np.save(Kv_diag_file, Kv_diag, allow_pickle=False)

    if Kt_diag_file is not None:
        if os.path.isfile(Kt_diag_file):
            print("Skipping Kt_diag")
        else:
            print("Computing Kt_diag")
            Kt_diag = compute_big_Kdiag(sess, kern, n_max=n_max*n_max,
                                        X=Xt, n_gpus=n_gpus)
            np.save(Kt_diag_file, Kt_diag, allow_pickle=False)


def main(_):
    FLAGS = tf.app.flags.FLAGS
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    path = FLAGS.path
    if path is None:
        raise ValueError("Please provide a value for `FLAGS.path`")

    def file_for(name, fmt):
        return os.path.join(path, "{}_{:02d}.{}".format(name, FLAGS.seed, fmt))
    params_file = file_for('params', 'pkl.gz')
    gram_file = file_for('gram', 'npy')
    Kxvx_file = file_for('Kxvx', 'npy')
    Kxtx_file = file_for('Kxtx', 'npy')

    # Replicate the parameters the experiments were ran with
    if FLAGS.allow_skip:
        LAYERS_MIN=4
        LAYERS_SPAN=12
        FILTER_SIZE_SPAN=4
    else:
        LAYERS_MIN=2
        LAYERS_SPAN=7
        FILTER_SIZE_SPAN=5

    if os.path.isfile(params_file):
        params = pu.load(params_file)
    else:
        params = dict(
            seed=FLAGS.seed,
            var_weight=np.random.rand() * 8 + 0.5,
            var_bias=np.random.rand() * 8 + 0.2,
            n_layers=LAYERS_MIN + int(np.random.rand()*LAYERS_SPAN),
            filter_sizes=3 + int(np.random.rand()*FILTER_SIZE_SPAN),
            strides=1 + int(np.random.rand()*3),
            padding=("VALID" if np.random.rand() > 0.5 else "SAME"),
            nlin=("ExReLU" if np.random.rand() > 0.5 else "ExErf"),
            skip_freq=(int(np.random.rand()*2) + 1
                       if ((np.random.rand() > 0.5 and FLAGS.allow_skip)
                           or FLAGS.seed < 56)  # Before that, skip_freq always positive
                       else -1),
        )
        if params['skip_freq'] > 0:
            params['padding'] = 'SAME'
            params['strides'] = 1

    print("Params:", sorted(list(params.items())))
    pu.dump(params, params_file)
    with tf.device("cpu:0"):
        kern = create_kern(params)

    save_kernels(kern, FLAGS.N_train, FLAGS.N_vali, FLAGS.n_gpus, gram_file,
                 Kxvx_file, Kxtx_file, n_max=FLAGS.n_max)


if __name__ == '__main__':
    f = tf.app.flags
    f.DEFINE_integer('n_gpus', 1, "Number of GPUs to use")
    f.DEFINE_boolean('allow_skip', False, "Whether to use skip connections in the random draw")
    f.DEFINE_integer('seed', 0, (
        "random seed (no randomness in this program, use to save different "
        "versions of the resnet kernel)"))
    f.DEFINE_integer('n_max', 400,
        "max number of examples to simultaneously compute the kernel of")
    f.DEFINE_integer('N_train', 50000, 'number of training data points')
    f.DEFINE_integer('N_vali', 10000, 'number of validation data points')
    f.DEFINE_string('path', None, "path to save kernel matrices to")
    tf.app.run()
