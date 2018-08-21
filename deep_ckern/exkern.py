from gpflow.params import Parameterized
import tensorflow as tf
import numpy as np

__all__ = ['ElementwiseExKern', 'ExReLU', 'ExErf']


class ElementwiseExKern(Parameterized):
    def K(self, cov, var1, var2=None):
        raise NotImplementedError

    def Kdiag(self, var):
        raise NotImplementedError

    def nlin(self, x):
        """
        The nonlinearity that this is computing the expected inner product of.
        Used for testing.
        """
        raise NotImplementedError


class ExReLU(ElementwiseExKern):
    def __init__(self, exponent=1, multiply_by_sqrt2=False, name=None):
        super(ExReLU, self).__init__(name=name)
        self.multiply_by_sqrt2 = multiply_by_sqrt2
        if exponent in {0, 1}:
            self.exponent = exponent
        else:
            raise NotImplementedError

    def K(self, cov, var1, var2=None):
        if var2 is None:
            sqrt1 = sqrt2 = tf.sqrt(var1)
        else:
            sqrt1, sqrt2 = tf.sqrt(var1), tf.sqrt(var2)

        norms_prod = sqrt1[:, None, ...] * sqrt2
        norms_prod = tf.reshape(norms_prod, tf.shape(cov))

        cos_theta = tf.clip_by_value(cov / norms_prod, -1., 1.)
        theta = tf.acos(cos_theta)  # angle wrt the previous RKHS

        if self.exponent == 0:
            return .5 - theta/(2*np.pi)

        sin_theta = tf.sqrt(1. - cos_theta**2)
        J = sin_theta + (np.pi - theta) * cos_theta
        if self.multiply_by_sqrt2:
            div = np.pi
        else:
            div = 2*np.pi
        return norms_prod / div * J

    def Kdiag(self, var):
        if self.multiply_by_sqrt2:
            if self.exponent == 0:
                return tf.ones_like(var)
            else:
                return var
        else:
            if self.exponent == 0:
                return tf.ones_like(var)/2
            else:
                return var/2

    def nlin(self, x):
        if self.multiply_by_sqrt2:
            if self.exponent == 0:
                return ((1 + tf.sign(x))/np.sqrt(2))
            elif self.exponent == 1:
                return tf.nn.relu(x) * np.sqrt(2)
        else:
            if self.exponent == 0:
                return ((1 + tf.sign(x))/2)
            elif self.exponent == 1:
                return tf.nn.relu(x)


class ExErf(ElementwiseExKern):
    """The Gaussian error function as a nonlinearity. It's very similar to the
    tanh. Williams 1997"""
    def K(self, cov, var1, var2=None):
        if var2 is None:
            t1 = t2 = 1+2*var1
        else:
            t1, t2 = 1+2*var1, 1+2*var2
        vs = tf.reshape(t1[:, None, ...] * t2, tf.shape(cov))
        sin_theta = 2*cov / tf.sqrt(vs)
        return (2/np.pi) * tf.asin(sin_theta)

    def Kdiag(self, var):
        v2 = 2*var
        return (2/np.pi) * tf.asin(v2 / (1 + v2))

    def nlin(self, x):
        return tf.erf(x)
