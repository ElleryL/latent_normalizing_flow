import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.layers import core as layers
from tensorflow.python.ops import template as template_ops

__all__ = ["vecToSkew","matrxExp", "Orthogonal_Flow","orthogonal_flow_template"]

class Orthogonal_Flow(bijector.Bijector):
    def __init__(self,
                 transformation,
                 special_group=True,
                 validate_args=False,
                 name=None):
        name = name or "orthogonal_flow"
        with tf.name_scope(name, name):
            self.special_group = special_group
            self.transformation = transformation
            self.out_dim = None
            super(Orthogonal_Flow, self).__init__(
                forward_min_event_ndims=1,
                inverse_min_event_ndims=1,
                is_constant_jacobian=True,
                validate_args=validate_args,
                name=name or "orthogonal")

    def _cache_input_depth(self, x):
        if self.out_dim is None:
            self.out_dim = x.shape.with_rank_at_least(1)[-1].value

    def _forward(self, x0):
        self._cache_input_depth(x0)
        x,y = self.transformation(x0,self.out_dim)

        # construct skew matrix
        A = vecToSkew(x, y) # batch_size x D x D

        # apply exponential
        O = matrxExp(A)
        if not self.special_group: # determinants of -1
            O = -O
        # perform permutation
        v = tf.matmul(O, x0[:,:,None])[:,:,0]
        return v

    def _inverse(self, y0):
        self._cache_input_depth(y0)
        x, y = self.transformation(y0, self.out_dim)

        # construct skew matrix
        A = vecToSkew(x, y)

        # apply log
        O = matrxExp(A)
        if not self.special_group: # determinants of -1
            O = -O
        # perform permutation
        invO = tf.linalg.inv(O)
        v = tf.matmul(invO, y0[:,:,None])[:,:,0]
        return v


    def _inverse_log_det_jacobian(self, y):
        '''
        orthogonal matrix has determinants -1 or 1
        the log of its abs is zero
        '''
        return 0.

    def _forward_log_det_jacobian(self, x):
        '''
        orthogonal matrix has determinants -1 or 1
        the log of its abs is zero
        '''
        return 0.

def orthogonal_flow_template(hidden_layers,
                             activation=nn_ops.relu,
                             name=None,
                             *args,
                             **kwargs):
    with ops.name_scope(name, "orthogonal_flow_template"):
        def _fn(x,output_units):
            for units in hidden_layers:
                x = layers.dense(
                    inputs=x,
                    units=units,
                    activation=activation,
                    # kernel_initializer=tf.random_normal_initializer(0., .01,seed=14),
                    # bias_initializer=tf.random_normal_initializer(0., .01,seed=14),
                    *args,
                    **kwargs)
            x = layers.dense(
                inputs=x,
                units=2 * output_units,
                activation=None,
                *args,
                **kwargs)
            x, y = array_ops.split(x, 2, axis=-1)
            return x,y
        return template_ops.make_template(
        "orthogonal_flow_template", _fn)

def vecToSkew(x,y):
    '''
    Take two vectors and construct a skew-symmetric matrix out of it
    return x.T*y - y.T*x
    :param x: vector
    :param y: vector
    :return: A Skew Symmetric Matrix
    '''
    A = tf.matmul(tf.expand_dims(x, -1), tf.expand_dims(y, 1))
    B = tf.matmul(tf.expand_dims(y, -1), tf.expand_dims(x, 1))
    result_tensor = A - B

    return result_tensor

def matrxLog(input,name=None):  # pylint: disable=redefined-builtin
  with ops.name_scope(name, 'log_matrix', [input]):
    return tf.cast(tf.linalg.logm(tf.cast(input, dtype=tf.complex64)), dtype=input.dtype)
    # matrix = ops.convert_to_tensor(input, name='input')
    # if matrix.shape[-2:] == [0, 0]:
    #   return matrix
    # batch_shape = matrix.shape[:-2]
    # if not batch_shape.is_fully_defined():
    #   batch_shape = array_ops.shape(matrix)[:-2]
    #
    # matrix = array_ops.reshape(
    #     matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-2:]), axis=0))
    #
    # # TODO: Implement a skew-symmetric version of this
    # base_logm = tf.linalg.logm(tf.cast(matrix,dtype=tf.complex64))
    # base_logm = tf.cast(base_logm,dtype=input.dtype)
    # avg_base_logm = 0.5*(base_logm + array_ops.transpose(base_logm,perm=[0,2,1]))
    # result = base_logm - avg_base_logm
    #
    # if not matrix.shape.is_fully_defined():
    #   return array_ops.reshape(
    #       result,
    #       array_ops.concat((batch_shape, array_ops.shape(result)[-2:]), axis=0))
    # return array_ops.reshape(result, batch_shape.concatenate(result.shape[-2:]))


def matrxExp(input, name=None):
    with ops.name_scope(name, 'exp_matrix', [input]):
        return tf.linalg.expm(input)
        # matrix = ops.convert_to_tensor(input, name='input')
        # if matrix.shape[-2:] == [0, 0]:
        #     return matrix
        # batch_shape = matrix.shape[:-2]
        # if not batch_shape.is_fully_defined():
        #     batch_shape = array_ops.shape(matrix)[:-2]
        #
        # # reshaping the batch makes the where statements work better
        # matrix = array_ops.reshape(
        #     matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-2:]), axis=0))
        # l1_norm = math_ops.reduce_max(
        #     math_ops.reduce_sum(math_ops.abs(matrix),
        #                         axis=array_ops.size(array_ops.shape(matrix)) - 2),
        #     axis=-1)
        # const = lambda x: constant_op.constant(x, l1_norm.dtype)
        #
        # if matrix.dtype in [dtypes.float16, dtypes.float32, dtypes.complex64]:
        #     maxnorm = const(3.925724783138660)
        #     squarings = math_ops.maximum(
        #         math_ops.floor(
        #             math_ops.log(l1_norm / maxnorm) / math_ops.log(const(2.0))), 0)
        # elif matrix.dtype in [dtypes.float64, dtypes.complex128]:
        #     maxnorm = const(5.371920351148152)
        #     squarings = math_ops.maximum(
        #         math_ops.floor(
        #             math_ops.log(l1_norm / maxnorm) / math_ops.log(const(2.0))), 0)
        #
        # else:
        #     raise ValueError(
        #         'skew_sym_expm does not support matrices of type %s' % matrix.dtype)
        #
        # B = matrix / math_ops.pow(
        #     constant_op.constant(2.0, dtype=matrix.dtype),
        #     math_ops.cast(squarings, matrix.dtype))[...,
        #                                             array_ops.newaxis,
        #                                             array_ops.newaxis]
        #
        # B1 = math_ops.matmul(B, B)
        # B2 = math_ops.matmul(B1, B1)
        # B3 = math_ops.matmul(B1, B2)
        #
        # ident = linalg_ops.eye(array_ops.shape(matrix)[-2],
        #                        batch_shape=array_ops.shape(matrix)[:-2],
        #                        dtype=matrix.dtype)
        # P1 = 17297280 * ident + 1995840 * B1 + 25200 * B2 + 56 * B3
        # P2 = 8648640 * B + math_ops.matmul(B, 277200 * B1 + 1512 * B2 + B3)
        #
        # result = linalg_ops.matrix_solve(P1 - P2, P1 + P2)
        # max_squarings = math_ops.reduce_max(squarings)
        #
        # i = const(0.0)
        # c = lambda i, r: math_ops.less(i, max_squarings)
        #
        # def b(i, r):
        #     return i + 1, array_ops.where(math_ops.less(i, squarings),
        #                                   math_ops.matmul(r, r), r)
        #
        # _, result = control_flow_ops.while_loop(c, b, [i, result])
        # if not matrix.shape.is_fully_defined():
        #     return array_ops.reshape(
        #         result,
        #         array_ops.concat((batch_shape, array_ops.shape(result)[-2:]), axis=0))
        # return array_ops.reshape(result, batch_shape.concatenate(result.shape[-2:]))
