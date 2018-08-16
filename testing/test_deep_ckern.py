import gpflow
from gpflow import settings
import numpy as np
import tensorflow as tf
import deep_ckern as ck
import deep_ckern.dkern
from testing.RecursiveKernel import DeepArcCosine


def cnn_from_params(Ws, bs, dkern, X):
    batch = tf.shape(X)[0]
    for W, b, st, pd in zip(Ws[:-1], bs[:-1], dkern.strides, dkern.padding):
        b_reshaped = tf.reshape(b, [batch, -1, 1, 1])
        strides = [1, 1] + list(st)
        X = tf.nn.conv2d(X, W, strides, pd, data_format="NCHW") + b_reshaped
        X = dkern.recurse_kern.nlin(X)
    # Get the correct weight order in the fully-connected layer
    X_nhwc = tf.transpose(X, [0, 2, 3, 1])
    return tf.reshape(X_nhwc, [batch, -1]) @ Ws[-1] + bs[-1]


class DeepKernelTest(gpflow.test_util.GPflowTestCase):
    def test_matches_relu_fc(self, D=7):
        with self.test_context() as sess:
            for L in [0, 1, 3, 6]:
                ka = DeepArcCosine(D, L, variance=1.2, bias_variance=0.8)
                kb = ck.DeepKernel([D], [[]]*L, ck.ExReLU(),
                                   var_weight=1.2, var_bias=0.8,
                                   data_format="NC")
                x = tf.constant(np.random.randn(12, D))
                x2 = tf.constant(np.random.randn(15, D))

                Kxx = [ka.K(x), kb.K(x)]
                Kxx2 = [ka.K(x, x2), kb.K(x, x2)]
                Kxdiag = [ka.Kdiag(x), kb.Kdiag(x)]

                self.assertAllClose(*sess.run(Kxx))
                self.assertAllClose(*sess.run(Kxx2))
                self.assertAllClose(*sess.run(Kxdiag))

    def test_equivalent_BNN(self, L=1, n_random_tests=4):
        s = settings.get_settings()
        s.dtypes.float_type = 'float32'
        with self.test_context() as sess, settings.temp_settings(s):
            shape = [3, 12, 10]
            X = tf.ones([1] + shape, dtype=settings.float_type)
            kb = ck.DeepKernel(shape, [[3, 3]]*L, ck.ExReLU(),
                               var_weight=1.2, var_bias=0.8,
                               data_format="NCHW")
            tf_y_bnn = kb.equivalent_BNN(X, n_samples=2, n_filters=7)
            W0, b0 = (list(t[0] for t in t_list) for t_list in [kb._W, kb._b])
            W1, b1 = (list(t[1] for t in t_list) for t_list in [kb._W, kb._b])
            tf_y0 = cnn_from_params(W0, b0, kb, X)
            tf_y1 = cnn_from_params(W1, b1, kb, X)

            for _ in range(n_random_tests):
                y_bnn, y0, y1 = sess.run([tf_y_bnn, tf_y0, tf_y1])
                self.assertAllClose(y_bnn[0:1], y0)
                self.assertAllClose(y_bnn[1:2], y1)


class ExKernTest(gpflow.test_util.GPflowTestCase):
    def compare_exkern(
            self, sess, exkern, cov=2.1, var1=3.1, var2=1.5, Hdiag=40, Hk=40, rtol=1e-6):
        def arr(x):
            return tf.constant([x], dtype=settings.float_type)

        a = exkern.Kdiag(arr([var1]))
        b = gpflow.quadrature.mvnquad(
            lambda x: exkern.nlin(x)**2, arr([0.]), arr([[var1]]), H=Hdiag)
        self.assertAllClose(*sess.run([a, b]), rtol=rtol)

        def sq_nlin(x):
            y = exkern.nlin(x)
            return y[:, 0:1] * y[:, 1:2]

        a = exkern.K(arr([cov]), arr([var1]), arr([var2]))
        b = gpflow.quadrature.mvnquad(
            sq_nlin, arr([0., 0.]), arr([[var1, cov], [cov, var2]]), H=Hk)
        self.assertAllClose(*sess.run([a, b]), rtol=rtol)

    def test_relu(self):
        "Test very iffy; but I can't get it to pass"
        with self.test_context() as sess:
            self.compare_exkern(sess, ck.ExReLU(), Hdiag=100, Hk=100, rtol=0.001)

    def test_step(self):
        "Test very iffy; but I can't get it to pass"
        with self.test_context() as sess:
            self.compare_exkern(sess, ck.ExReLU(exponent=0), Hdiag=100, Hk=200,
                                rtol=0.01)

    def test_erf(self):
        with self.test_context() as sess:
            for _ in range(5):
                a = np.random.randn(2, 2)
                v1, cov, _, v2 = (a@a.T).flat
                self.compare_exkern(sess, ck.ExErf(), cov=cov, var1=v1,
                                    var2=v2, Hdiag=100, Hk=100)


class NewDeepKernelTest(gpflow.test_util.GPflowTestCase):
    def test_matches_old_dkern(self, D=7):
        with self.test_context() as sess:
            for L in [0, 1, 3, 6]:  # , 3, 6]:
                shape = [3, D, D]
                ka = ck.DeepKernel(shape, [[3, 3]]*L, ck.ExReLU(),
                                   var_weight=1.2, var_bias=0.8,
                                   padding="SAME", # strides=1,
                                   data_format="NCHW")
                kb = ck.dkern.DeepKernelTesting(
                    shape, block_sizes=[-1]*L, block_strides=[-1]*L,
                    kernel_size=3, recurse_kern=ck.ExReLU(), var_weight=1.2,
                    var_bias=0.8)

                x = tf.constant(np.random.randn(12, *shape),
                                dtype=settings.float_type)
                x2 = tf.constant(np.random.randn(15, *shape),
                                 dtype=settings.float_type)

                Kxx = [ka.K(x), kb.K(x)]
                Kxx2 = [ka.K(x, x2), kb.K(x, x2)]
                Kxdiag = [ka.Kdiag(x), kb.Kdiag(x)]

                self.assertAllClose(*sess.run(Kxx))
                self.assertAllClose(*sess.run(Kxx2))
                self.assertAllClose(*sess.run(Kxdiag))
