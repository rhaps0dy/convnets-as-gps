"""
Using a precomputed kernel matrix, compute test scores of GP regression
"""
import sys
import pickle_utils as pu
import numpy as np
import scipy
import sklearn.metrics
import time
import pandas as pd
import os
from save_kernels import create_kern, compute_big_K

from absl import app as absl_app
from absl import flags

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


def cut_training(N, Kxx, Kxvx, Kv_diag, Kxtx, Kt_diag, X, Y, Xv, Yv, Xt, Yt):
    return (Kxx[:N, :N],
            np.concatenate([Kxx[N:, :N], Kxvx[:, :N]], axis=0),
            np.concatenate([np.diag(Kxx[N:, N:]), Kv_diag], axis=0),
            Kxtx[:, :N],
            Kt_diag,
            X[:N], Y[:N],
            np.concatenate([X[N:], Xv], axis=0),
            np.concatenate([Y[N:], Yv], axis=0),
            Xt, Yt)


def main(_):
    start_time = time.time()
    FLAGS = flags.FLAGS
    seed = FLAGS.seed
    path = FLAGS.path

    # noise = float(sys.argv[2])
    params_file = os.path.join(path, "params_{:02d}.pkl.gz".format(seed))
    gram_file = os.path.join(path, "gram_{:02d}.npy".format(seed))
    Kxvx_file = os.path.join(path, "Kxvx_{:02d}.npy".format(seed))
    Kxtx_file = os.path.join(path, "Kxtx_{:02d}.npy".format(seed))
    Kv_diag_file = os.path.join(path, "Kv_diag_{:02d}.npy".format(seed))
    Kt_diag_file = os.path.join(path, "Kt_diag_{:02d}.npy".format(seed))
    csv_file = os.path.join(path, FLAGS.csv_dir, "{:02d}.csv".format(seed))

    params = pu.load(params_file)
    print("Kernel params:", params)

    # Kxtx = np.load(Kxtx_file)
    # if Kxtx.shape[1] != 10000:
    #     params['test_error'] = -1.
    #     params['validation_error'] = -1.
    #     params['training_error'] = -1.
    #     params['time'] = -1.
    #     pd.DataFrame(data=params, index=pd.Index([0])).to_csv(csv_file)
    #     print("Test is wrong size for seed", seed)
    #     sys.exit(1)

    print("Loading data and kernels")
    Kxx, Kxvx, Kv_diag, Kxtx, Kt_diag, X, Y, Xv, Yv, Xt, Yt = cut_training(
        FLAGS.N, np.load(gram_file),
        np.load(Kxvx_file), np.load(Kv_diag_file),
        np.load(Kxtx_file), np.load(Kt_diag_file),
        *mnist_1hot_all())

    # center labels
    Kxx = Kxx.astype(np.float64)
    Y[Y == 0.] = -1

    print("Solving system")
    print("Size of Kxx, Y, Kxvx, Yv, Kxtx, Yt:", Kxx.shape, Y.shape, Kxvx.shape, Yv.shape, Kxtx.shape, Yt.shape)
    K_inv_y = scipy.linalg.solve(Kxx, Y, overwrite_a=True, overwrite_b=False, check_finite=False, assume_a='pos')
    #K_inv_y = Kxx @ Y
    #print("Saving K_inv_y...")
    #np.save(sys.argv[1].replace('gram_', 'K_inv_y_'), K_inv_y)
    #ys.exit(0)

    def print_error(K_xt_x, Ytv, dit, key):
        print("Running for", key)
        Y_pred = K_xt_x @ K_inv_y
        t = sklearn.metrics.accuracy_score(
            np.argmax(Ytv, 1), np.argmax(Y_pred, 1))
        dit[key] = (1-t)*100

    print_error(Kxx, Y, params, "training_error")
    print_error(Kxvx, Yv, params, "validation_error")
    print_error(Kxtx, Yt, params, "test_error")
    params['time'] = time.time() - start_time
    pd.DataFrame(data=params, index=pd.Index([0])).to_csv(csv_file)

if __name__ == '__main__':
    flags.DEFINE_integer('seed', None, 'random seed')
    flags.DEFINE_integer('n_gpus', 1, 'n of gpus')
    flags.DEFINE_integer('N', 50000, 'number of data points')
    flags.DEFINE_string('path', "./precomputed_kernels", "the path to the precomputed kernel matrices")
    flags.DEFINE_string('csv_dir', 'dfs', 'directory to save csvs, under gram matrix path')
    absl_app.run(main)
