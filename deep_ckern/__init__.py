import gpflow
from gpflow import settings
from gpflow.params import Parameter, Parameterized, DataHolder, Minibatch
import tensorflow as tf
import numpy as np
from typing import List

from .exkern import ElementwiseExKern, ExReLU, ExErf
from .resnet import ResnetKernel


class DeepKernel(gpflow.kernels.Kernel):
    "General deep kernel for n-dimensional convolutional networks"
    def __init__(self,
                 input_shape: List[int],
                 filter_sizes: List[List[int]],
                 recurse_kern: ElementwiseExKern,
                 var_weight: float = 1.0,
                 var_bias: float = 1.0,
                 padding: List[str] = "SAME",
                 strides: List[List[int]] = None,
                 data_format: str = "NCHW",
                 active_dims: slice = None,
                 skip_freq: int = -1,
                 name: str = None):
        input_dim = np.prod(input_shape)
        super(DeepKernel, self).__init__(input_dim, active_dims, name=name)

        self.filter_sizes = np.copy(filter_sizes).astype(np.int32)
        self.n_layers = len(filter_sizes)
        self.input_shape = list(np.copy(input_shape))
        self.recurse_kern = recurse_kern
        self.skip_freq = skip_freq

        inferred_data_format = "NC" + "DHW"[4-len(input_shape):]
        if inferred_data_format != data_format:
            raise ValueError(("Inferred and supplied data formats "
                              "inconsistent: {} vs {}")
                             .format(data_format, inferred_data_format))
        self.data_format = data_format

        if not isinstance(padding, list):
            self.padding = [padding] * len(self.filter_sizes)
        else:
            self.padding = padding
        if len(self.padding) != len(self.filter_sizes):
            raise ValueError(("Mismatching number of layers in `padding` vs "
                              "`filter_sizes`: {} vs {}").format(
                                  len(self.padding), len(self.filter_sizes)))

        if strides is None:
            self.strides = np.ones([self.n_layers, len(input_shape)-1],
                                   dtype=np.int32)
        else:
            self.strides = np.copy(strides).astype(np.int32)
        if len(self.strides) != self.n_layers:
            raise ValueError(("Mismatching number of layers in `strides`: "
                              "{} vs {}").format(
                                  len(self.strides), self.n_layers))

        self.var_weight = Parameter(var_weight, gpflow.transforms.positive)
        self.var_bias = Parameter(var_bias, gpflow.transforms.positive)

    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def K(self, X, X2=None):
        # Concatenate the covariance between X and X2 and their respective
        # variances. Only 1 variance is needed if X2 is None.
        if X2 is None:
            N = N2 = tf.shape(X)[0]
            var_z_list = [
                tf.reshape(tf.square(X), [N] + self.input_shape),
                tf.reshape(X[:, None, :] * X, [N*N] + self.input_shape)]
            cross_start = N
        else:
            N, N2 = tf.shape(X)[0], tf.shape(X2)[0]
            var_z_list = [
                tf.reshape(tf.square(X), [N] + self.input_shape),
                tf.reshape(tf.square(X2), [N2] + self.input_shape),
                tf.reshape(X[:, None, :] * X2, [N*N2] + self.input_shape)]
            cross_start = N + N2
        var_z_list = [tf.reduce_mean(z, axis=1, keepdims=True)
                      for z in var_z_list]
        var_z_previous = None

        for i in range(self.n_layers):
            # Do the convolution for all the co/variances at once
            var_z = tf.concat(var_z_list, axis=0)
            if ((isinstance(self.skip_freq, list) and i in self.skip_freq) or (
                    self.skip_freq > 0 and i % self.skip_freq == 0)):
                var_z = var_z + var_z_previous
                var_z_previous = var_z
            if i == 0:
                # initialize var_z_previous
                var_z_previous = var_z
            var_a_all = self.lin_step(i, var_z)

            # Disentangle the output of the convolution and compute the next
            # layer's co/variances
            var_a_cross = var_a_all[cross_start:]
            if X2 is None:
                var_a_1 = var_a_all[:N]
                var_z_list = [self.recurse_kern.Kdiag(var_a_1),
                              self.recurse_kern.K(var_a_cross, var_a_1, None)]
            else:
                var_a_1 = var_a_all[:N]
                var_a_2 = var_a_all[N:cross_start]
                var_z_list = [self.recurse_kern.Kdiag(var_a_1),
                              self.recurse_kern.Kdiag(var_a_2),
                              self.recurse_kern.K(var_a_cross, var_a_1, var_a_2)]
        # The final layer
        var_z_cross = tf.reshape(var_z_list[-1], [N, N2, -1])
        var_z_cross_last = tf.reduce_mean(var_z_cross, axis=2)
        return self.var_bias + self.var_weight * var_z_cross_last

    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def Kdiag(self, X):
        X_sq = tf.reshape(tf.square(X), [-1] + self.input_shape)
        var_z = tf.reduce_mean(X_sq, axis=1, keepdims=True)
        for i in range(self.n_layers):
            var_a = self.lin_step(i, var_z)
            var_z = self.recurse_kern.Kdiag(var_a)

        all_except_first = np.arange(1, len(var_z.shape))
        var_z_last = tf.reduce_mean(var_z, axis=all_except_first)
        return self.var_bias + self.var_weight * var_z_last

    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def lin_step(self, i, x):
        if len(x.shape) == 2:
            a = self.var_weight * x
        else:
            f = tf.fill(list(self.filter_sizes[i]) + [1, 1], self.var_weight)
            a = tf.nn.convolution(
                x, f, padding=self.padding[i], strides=self.strides[i],
                data_format=self.data_format)
            # a = tf.nn.conv2d(
            #     x, f,
            #     strides=[1]+self.strides[i]+[1],
            #     padding=self.padding[i],
            #     data_format='NCHW')
        return a + self.var_bias


    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def get_Wb(self, i, X_shape=None, n_samples=None, n_filters=None):
        "Unlike the kernel, this operates in NHWC"
        try:
            if self._W[i] is not None and self._b[i] is not None:
                return self._W[i], self._b[i]
        except AttributeError:
            self._W, self._b = ([None]*(self.n_layers + 1) for _ in 'Wb')
        try:
            std_b = self._std_b
        except AttributeError:
            std_b = self._std_b = tf.sqrt(self.var_bias)

        if i == self.n_layers:  # Final weights and biases
            final_dim = np.prod(list(map(int, X_shape[1:])))
            shape_W = [n_samples, final_dim, n_filters]
            shape_b = [n_samples, n_filters]
            std_W = tf.sqrt(self.var_weight / final_dim)
        else:
            if i == 0:
                fan_in = int(X_shape[-1])
            else:
                fan_in = n_filters
            fs = list(self.filter_sizes[i])
            shape_W = [n_samples] + fs + [fan_in, n_filters]
            shape_b = [n_samples] + [1]*len(fs) + [n_filters]
            std_W = tf.sqrt(self.var_weight / fan_in)

        self._W[i] = tf.random_normal(shape_W, stddev=std_W,
                                      name="W_{}".format(i), dtype=settings.float_type)
        self._b[i] = tf.random_normal(shape_b, stddev=std_b,
                                      name="b_{}".format(i), dtype=settings.float_type)
        return self._W[i], self._b[i]


    def fast_1sample_equivalent_BNN(self, X, Ws=None, bs=None):
        if Ws is None or bs is None:
            Ws, bs = (list(t[0] for t in t_list) for t_list in [self._W, self._b])
        batch = tf.shape(X)[0]
        for W, b, st, pd in zip(Ws[:-1], bs[:-1], self.strides, self.padding):
            b_reshaped = tf.reshape(b, [1, -1, 1, 1])
            strides = [1, 1] + list(st)
            X = tf.nn.conv2d(X, W, strides, pd, data_format="NCHW") + b_reshaped
            X = self.recurse_kern.nlin(X)
        return tf.reshape(X, [batch, -1]) @ Ws[-1] + bs[-1]


    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def equivalent_BNN(self, X, n_samples, n_filters=128):
        if list(map(int, X.shape)) != [1] + self.input_shape:
            raise NotImplementedError("Can only deal with 1 input image")

        # Unlike the kernel, this function operates in NHWC. This is because of
        # the `extract_image_patches` function
        tp_order = np.concatenate([[0], np.arange(2, len(X.shape)), [1]])
        X = tf.transpose(X, tp_order)  # NCHW -> NHWC

        # The name of the first dimension of the einsum. In the first linear
        # transform, it should be "a", to broadcast the "n" dimension of
        # samples of parameters along it. In all other iterations it should be
        # "n".
        first = 'a'
        batch_dim = 1

        for i in range(self.n_layers):
            if len(self.filter_sizes[i]) == 0:
                Xp = X
            elif len(self.filter_sizes[i]) == 2:
                h, w = self.filter_sizes[i]
                sh, sw = self.strides[i]
                Xp = tf.extract_image_patches(
                    X, [1, h, w, 1], [1, sh, sw, 1], [1, 1, 1, 1],
                    self.padding[i])
            else:
                raise NotImplementedError("convolutions other than 2d")

            W, b = self.get_Wb(i, X.shape, n_samples, n_filters)
            equation = "{first:}{dims:}i,nij->n{dims:}j".format(
                first=first, dims="dhw"[4-len(self.input_shape):])

            # We're explicitly doing the convolution by extracting patches and
            # a multiplication, so this flatten is needed.
            W_flat_in = tf.reshape(W, [n_samples, -1, W.shape[-1]])
            X = self.recurse_kern.nlin(tf.einsum(equation, Xp, W_flat_in) + b)
            first = 'n'  # Now we have `n_samples` in the batch dimension
            batch_dim = n_samples

        W, b = self.get_Wb(self.n_layers, X.shape, n_samples, 1)
        X_flat = tf.reshape(X, [batch_dim, -1])
        Wx = tf.einsum("{first:}i,nij->nj".format(first=first), X_flat, W)
        return Wx + b



class ZeroMeanGauss(gpflow.priors.Gaussian):
    def __init__(self, var):
        gpflow.priors.Prior.__init__(self)
        self.mu = 0.0
        self.var = var

    def logp(self, x):
        c = np.log(2*np.pi) + np.log(self.var)
        return -.5 * (c*tf.cast(tf.size(x), settings.float_type)
                      + tf.reduce_sum(tf.square(x)/self.var))


class ConvNet(gpflow.models.Model):
    "L2-regularised ConvNet as a Model"
    def __init__(self, X, Y, kern, minibatch_size=None, n_filters=256, name: str = None):
        super(ConvNet, self).__init__(name=name)
        if not hasattr(kern, 'W_'):
            # Create W_ and b_ as attributes in kernel
            X_zeros = np.zeros([1] + kern.input_shape)
            _ = kern.equivalent_BNN(
                X=tf.constant(X_zeros, dtype=settings.float_type),
                n_samples=1,
                n_filters=n_filters)
        self._kern = kern

        # Make MiniBatches if necessary
        if minibatch_size is None:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y, dtype=tf.int32)
            self.scale_factor = 1.
        else:
            self.X = Minibatch(X, batch_size=minibatch_size, seed=0)
            self.Y = Minibatch(Y, batch_size=minibatch_size, seed=0, dtype=np.int32)
            self.scale_factor = X.shape[0] / minibatch_size
        self.n_labels = int(np.max(Y)+1)

        # Create GPFlow parameters with the relevant size of the network
        Ws, bs = [], []
        for i, (W, b) in enumerate(zip(kern._W, kern._b)):
            if i == kern.n_layers:
                W_shape = [int(W.shape[1]), self.n_labels]
                b_shape = [self.n_labels]
            else:
                W_shape = list(map(int, W.shape[1:]))
                b_shape = [n_filters]
            W_var = kern.var_weight.read_value()/W_shape[-2]
            b_var = kern.var_bias.read_value()
            W_init = np.sqrt(W_var) * np.random.randn(*W_shape)
            b_init = np.sqrt(b_var) * np.random.randn(*b_shape)
            Ws.append(gpflow.params.Parameter(W_init, dtype=settings.float_type)) #, prior=ZeroMeanGauss(W_var)))
            bs.append(gpflow.params.Parameter(b_init, dtype=settings.float_type)) #, prior=ZeroMeanGauss(b_var)))
        self.Ws = gpflow.params.ParamList(Ws)
        self.bs = gpflow.params.ParamList(bs)

    def _build_objective(self, likelihood_tensor, prior_tensor):
        return self.scale_factor * likelihood_tensor - prior_tensor  # likelihood_tensor is already a loss

    @gpflow.decors.params_as_tensors
    def _build_likelihood(self):
        # Get around fast_1sample_equivalent_BNN not getting tensors from param
        Ws_tensors = list(self.Ws[i] for i in range(len(self.Ws)))
        bs_tensors = list(self.bs[i] for i in range(len(self.bs)))
        logits = self._kern.fast_1sample_equivalent_BNN(
            tf.reshape(self.X, [-1] + self._kern.input_shape),
            Ws=Ws_tensors, bs=bs_tensors)
        return tf.losses.sparse_softmax_cross_entropy(labels=self.Y, logits=logits)

    @gpflow.decors.autoflow((settings.float_type, [None, None]))
    def predict_y(self, Xnew):
        return self._build_predict_y(Xnew), tf.constant(0.0, dtype=settings.float_type)

    @gpflow.decors.params_as_tensors
    def _build_predict_y(self, Xnew):
        Ws_tensors = list(self.Ws[i] for i in range(len(self.Ws)))
        bs_tensors = list(self.bs[i] for i in range(len(self.bs)))
        logits = self._kern.fast_1sample_equivalent_BNN(
            tf.reshape(Xnew, [-1] + self._kern.input_shape),
            Ws=Ws_tensors, bs=bs_tensors)
        return tf.nn.softmax(logits)


if __name__ == '__main__':
    import deep_ckern as dk, numpy as np
    k = dk.DeepKernel([1, 16, 16], [[3, 3]]*5, dk.ExReLU())
    X = np.random.randn(3, 16**2)
    k.compute_K(X, X)
