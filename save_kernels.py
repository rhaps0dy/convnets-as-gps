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
from rectangles import mnist_1hot_all
import sys
import os
import gpflow


def create_kern(ps):
    if ps['seed'] == 1234:
        print('good shit')
        return dkern.DeepKernel(
            [1, 28, 28],
            filter_sizes=[[5, 5], [2, 2], [5, 5], [2, 2]],
            recurse_kern=dkern.ExReLU(),
            var_weight=1.,
            var_bias=1.,
            padding=["VALID", "SAME", "VALID", "SAME"],
            strides=[[1, 1]] * 4,
            data_format="NCHW",
            skip_freq=-1,
        )

    if 'skip_freq' not in ps:
        ps['skip_freq'] = -1
    return dkern.DeepKernel(
        [1, 28, 28],
        filter_sizes=[[ps['filter_sizes'], ps['filter_sizes']]] * ps['n_layers'],
        recurse_kern=getattr(dkern, ps['nlin'])(),
        var_weight=ps['var_weight'],
        var_bias=ps['var_bias'],
        padding=ps['padding'],
        strides=[[ps['strides'], ps['strides']]] * ps['n_layers'],
        data_format="NCHW",
        skip_freq=ps['skip_freq'],
    )


def compute_big_Kdiag(sess, kern, n_max, X, n_gpus=1):
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


def compute_big_K(sess, kern, n_max, X, X2=None, n_gpus=1,
                  use_Kdiag=False):
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


def save_kernels(kern, n_gpus, gram_file, Kxvx_file, Kxtx_file, n_max=400,
                 Kv_diag_file=None, Kt_diag_file=None):
    X, _, Xv, _, Xt, _ = mnist_1hot_all()
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


if __name__ == '__main__':
    seed = int(sys.argv[1])
    np.random.seed(seed)
    tf.set_random_seed(seed)

    path = "/scratch/ag919/grams/"
    params_file = os.path.join(path, "params_{:02d}.pkl.gz".format(seed))
    gram_file = os.path.join(path, "gram_{:02d}.npy".format(seed))
    Kxvx_file = os.path.join(path, "Kxvx_{:02d}.npy".format(seed))
    Kxtx_file = os.path.join(path, "Kxtx_{:02d}.npy".format(seed))
    n_gpus = 6

    if os.path.isfile(params_file):
        params = pu.load(params_file)
    else:
        params = dict(
            seed=seed,
            var_weight=np.random.rand() * 8 + 0.5,
            var_bias=np.random.rand() * 8 + 0.2,
            n_layers=4 + int(np.random.rand()*12),
            filter_sizes=3 + int(np.random.rand()*4),
            strides=1 + int(np.random.rand()*3),
            padding=("VALID" if np.random.rand() > 0.5 else "SAME"),
            nlin=("ExReLU" if np.random.rand() > 0.5 else "ExErf"),
            skip_freq=(int(np.random.rand()*2) + 1 if np.random.rand() > 0.5 else -1),
        )
        if params['skip_freq'] > 0:
            params['padding'] = 'SAME'
            params['strides'] = 1

    print("Params:", params)
    pu.dump(params, params_file)
    with tf.device("cpu:0"):
        kern = create_kern(params)

    save_kernels(kern, n_gpus, gram_file, Kxvx_file, Kxtx_file)
