"""
tensorflow/keras utilities for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

or for the transformation/integration functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# third party
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras import backend as K
# from keras.legacy import interfaces
# from tensorflow.keras.optimizers.legacy import interfaces
# from tensorflow import keras
from tensorflow.keras.layers import Layer, InputLayer, Input
from keras.engine.topology import Node


# local
from .utils import transform, resize, integrate_vec, affine_to_shift


class SpatialTransformer(Layer):
    """
    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle both affine and dense transforms. 
    Both transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel,
    and an affine transform gives the *difference* of the affine matrix from 
    the identity matrix.

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which 
    was in turn transformed to be dense with the help of (affine) STN code 
    via https://github.com/kevinzakka/spatial-transformer-network

    Since then, we've re-written the code to be generalized to any 
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        # if len(input_shape) > 2:
        #     raise Exception('Spatial Transformer must be called on a list of length 2.'
        #                     'First argument is the image, second is the transform.')
        
        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        # the transform is an affine iff:
        # it's a 1D Tensor [dense transforms need to be at least ndims + 1]
        # it's a 2D Tensor and shape == [N+1, N+1]. 
        #   [dense with N=1, which is the only one that could have a transform shape of 2, would be of size Mx1]
        self.is_affine = len(trf_shape) == 1 or \
                         (len(trf_shape) == 2 and all([f == (self.ndims+1) for f in trf_shape]))

        # check sizes
        if self.is_affine and len(trf_shape) == 1:
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise Exception('Expected flattened affine of len %d but got %d'
                                % (ex, trf_shape[0]))

        if not self.is_affine:
            if trf_shape[-1] != self.ndims:
                raise Exception('Offset flow field size expected: %d, found: %d' 
                                % (self.ndims, trf_shape[-1]))

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        # assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs) #######============
        vol = inputs[0]
        trf = inputs[1]

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        # go from affine
        if self.is_affine:
            trf = tf.map_fn(lambda x: self._single_aff_to_shift(x, vol.shape[1:-1]), trf, dtype=tf.float32)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # =================>weighted_adaptive
        if len(inputs)>2:
            wvol=inputs[2]
            # map transform across batch
            if self.single_transform:
                fn = lambda x: self._single_transform([x, trf[0, :],wvol])
                return tf.map_fn(fn, vol, dtype=tf.float32)
            else:
                return tf.map_fn(self._single_transform, [vol, trf, wvol], dtype=tf.float32)
        else:
            wvol= None
        # =================<weighted_adaptive
            # map transform across batch
            if self.single_transform:
                fn = lambda x: self._single_transform([x, trf[0,:]])
                return tf.map_fn(fn, vol, dtype=tf.float32)
            else:
                return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_aff_to_shift(self, trf, volshape):
        if len(trf.shape) == 1:  # go from vector to matrix
            trf = tf.reshape(trf, [self.ndims, self.ndims + 1])

        # note this is unnecessarily extra graph since at every batch entry we have a tf.eye graph
        trf += tf.eye(self.ndims+1)[:self.ndims,:]  # add identity, hence affine is a shift from identitiy
        return affine_to_shift(trf, volshape, shift_center=True)

    def _single_transform(self, inputs):
        # ================================>weight_adptive
        if len(inputs)>2:
            return transform(inputs[0], inputs[1], interp_method=self.interp_method,weight_vol=inputs[2])
        else:
            # ===================================<
            return transform(inputs[0], inputs[1], interp_method=self.interp_method)


class Resize(Layer):
    """
    N-D Resize Tensorflow / Keras Layer
    Note: this is not re-shaping an existing volume, but resizing, like scipy's "Zoom"

    If you find this function useful, please cite:
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,Dalca AV, Guttag J, Sabuncu MR
        CVPR 2018  

    Since then, we've re-written the code to be generalized to any 
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 zoom_factor,
                 interp_method='linear',
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        super(Resize, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        input_shape should be an element of list of one inputs:
        input1: volume
                should be a *vol_shape x N
        """

        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('Resize must be called on a list of length 1.'
                            'First argument is the image, second is the transform.')

        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        # set up number of dimensions
        self.ndims = len(input_shape) - 2
        self.inshape = input_shape

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: volume of list with one volume
        """

        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            vol = inputs[0]
        else:
            vol = inputs

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[1:]])

        # map transform across batch
        return tf.map_fn(self._single_resize, vol, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0]]
        output_shape += [int(f * self.zoom_factor) for f in input_shape[1:-1]]
        output_shape += [input_shape[-1]]
        return tuple(output_shape)

    def _single_resize(self, inputs):
        return resize(inputs, self.zoom_factor, interp_method=self.interp_method)

# Zoom naming of resize, to match scipy's naming
Zoom = Resize


class VecInt(Layer):
    """
    Vector Integration Layer

    Enables vector integration via several methods 
    (ode or quadrature for time-dependent vector fields, 
    scaling and squaring for stationary fields)

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self, indexing='ij', method='ss', int_steps=7, **kwargs):
        """        
        Parameters:
            method can be any of the methods in neuron.utils.integrate_vec
        """

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.method = method
        self.int_steps = int_steps
        self.inshape = None
        super(self.__class__, self).__init__(**kwargs)

    def build(self, input_shape):
        # confirm built
        self.built = True
        self.inshape = input_shape

        if input_shape[-1] != len(input_shape) - 2:
            raise Exception('transform ndims %d does not match expected ndims %d' \
                % (input_shape[-1], len(input_shape) - 2))

    def call(self, inputs):
        loc_shift = inputs

        # necessary for multi_gpu models...
        loc_shift = K.reshape(loc_shift, [-1, *self.inshape[1:]])
        loc_shift._keras_shape = inputs._keras_shape
        
        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            loc_shift_split = tf.split(loc_shift, loc_shift.shape[-1], axis=-1)
            loc_shift_lst = [loc_shift_split[1], loc_shift_split[0], *loc_shift_split[2:]]
            loc_shift = tf.concat(loc_shift_lst, -1)

        # map transform across batch
        out = tf.map_fn(self._single_int, loc_shift, dtype=tf.float32)
        out._keras_shape = inputs._keras_shape
        return out

    def _single_int(self, inputs):

        vel = inputs
        return integrate_vec(vel, method=self.method,
                      nb_steps=self.int_steps,
                      ode_args={'rtol':1e-6, 'atol':1e-12},
                      time_pt=1)
        

class LocalBias(Layer):
    """ 
    Local bias layer: each pixel/voxel has its own bias operation (one parameter)
    out[v] = in[v] + b
    """

    def __init__(self, my_initializer='RandomNormal', biasmult=1.0, **kwargs):
        self.initializer = my_initializer
        self.biasmult = biasmult
        super(LocalBias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalBias, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x + self.kernel * self.biasmult  # weights are difference from input

    def compute_output_shape(self, input_shape):
        return input_shape


# class LocalParam(InputLayer):

#     def __init__(self, shape, mult=1, my_initializer='RandomNormal', **kwargs):
#         super(LocalParam, self).__init__(input_shape=shape, **kwargs)       
       
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel', 
#                                       shape=tuple(shape),
#                                       initializer=my_initializer,
#                                       trainable=True)
        
#         outputs = self._inbound_nodes[0].output_tensors
#         z = Input(tensor=K.expand_dims(self.kernel, 0)*mult)
#         if len(outputs) == 1:
#             self._inbound_nodes[0].output_tensors[0] = z
#         else:
#             self._inbound_nodes[0].output_tensors = z
      
#     def get_output(self):  # call() would force inputs
#             outputs = self._inbound_nodes[0].output_tensors
#             if len(outputs) == 1:
#                 return outputs[0]
#             else:
#                 return outputs


#
# class LocalParam_new(Layer):
#
#     def __init__(self,
#                  shape,
#                  my_initializer='RandomNormal',
#                  name=None,
#                  mult=1.0,
#                  **kwargs):
#
#         self.shape = tuple([1, *shape])
#         self.my_initializer = my_initializer
#         self.mult = mult
#
#         super(LocalParam_new, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=tuple(self.shape[1:]),
#                                       initializer='uniform',
#                                       trainable=True)
#         super(LocalParam_new, self).build(input_shape)  # Be sure to call this at the end
#
#     def call(self, _):
#         # make sure it has a shape
#         if self.shape is not None:
#             self.kernel = tf.reshape(self.kernel, self.shape)
#         return self.kernel
#
#     def compute_output_shape(self, input_shape):
#         if self.shape is None:
#             return input_shape
#         else:
#             return self.shape
#
#
# class LocalParam(Layer):
#     """
#     Local Parameter layer: each pixel/voxel has its own parameter (one parameter)
#     out[v] = b
#
#     using code from
#     https://github.com/YerevaNN/R-NET-in-Keras/blob/master/layers/SharedWeight.py
#     and
#     https://github.com/keras-team/keras/blob/ee02d256611b17d11e37b86bd4f618d7f2a37d84/keras/engine/input_layer.py
#     """
#
#     def __init__(self,
#                  shape,
#                  my_initializer='RandomNormal',
#                  name=None,
#                  mult=1.0,
#                  **kwargs):
#         self.shape = [1, *shape]
#         self.my_initializer = my_initializer
#         self.mult = mult
#
#         if not name:
#             prefix = 'param'
#             name = '%s_%d' % (prefix, K.get_uid(prefix))
#         Layer.__init__(self, name=name, **kwargs)
#
#         # Create a trainable weight variable for this layer.
#         with K.name_scope(self.name):
#             self.kernel = self.add_weight(name='kernel',
#                                             shape=self.shape,
#                                             initializer=self.my_initializer,
#                                             trainable=True)
#
#         # prepare output tensor, which is essentially the kernel.
#         output_tensor = self.kernel * self.mult
#         output_tensor._keras_shape = self.shape
#         output_tensor._uses_learning_phase = False
#         output_tensor._keras_history = (self, 0, 0)
#         output_tensor._batch_input_shape = self.shape
#
#         self.trainable = True
#         self.built = True
#         self.is_placeholder = False
#
#         # create new node
#         Node(self,
#             inbound_layers=[],
#             node_indices=[],
#             tensor_indices=[],
#             input_tensors=[],
#             output_tensors=[output_tensor],
#             input_masks=[],
#             output_masks=[None],
#             input_shapes=[],
#             output_shapes=[self.shape])
#
#     def get_config(self):
#         config = {
#             '_batch_input_shape': self.shape,
#             '_keras_shape': self.shape,
#             'name': self.name
#         }
#         return config
#
#     def call(self, _):
#         z = self.get_output()
#         return tf.reshape(z, self.shape)
#
#     def compute_output_shape(self, input_shape):
#         return tuple(self.shape)
#
#     def get_output(self):  # call() would force inputs
#         outputs = self._inbound_nodes[0].output_tensors
#         if len(outputs) == 1:
#             return outputs[0]
#         else:
#             return outputs
#
#
# class MeanStream(Layer):
#     """
#     Maintain stream of data mean.
#
#     cap refers to mainting an approximation of up to that number of subjects -- that is,
#     any incoming datapoint will have at least 1/cap weight.
#     """
#
#     def __init__(self, cap=100, **kwargs):
#         self.cap = K.variable(cap, dtype='float32')
#         super(MeanStream, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Create mean and count
#         # These are weights because just maintaining variables don't get saved with the model, and we'd like
#         # to have these numbers saved when we save the model.
#         # But we need to make sure that the weights are untrainable.
#         self.mean = self.add_weight(name='mean',
#                                       shape=input_shape[1:],
#                                       initializer='zeros',
#                                       trainable=False)
#         self.count = self.add_weight(name='count',
#                                       shape=[1],
#                                       initializer='zeros',
#                                       trainable=False)
#
#         # self.mean = K.zeros(input_shape[1:], name='mean')
#         # self.count = K.variable(0.0, name='count')
#         super(MeanStream, self).build(input_shape)  # Be sure to call this somewhere!
#
#     def call(self, x):
#         # previous mean
#         pre_mean = self.mean
#
#         # compute this batch stats
#         this_sum = tf.reduce_sum(x, 0)
#         this_bs = tf.cast(K.shape(x)[0], 'float32')  # this batch size
#
#         # increase count and compute weights
#         new_count = self.count + this_bs
#         alpha = this_bs/K.minimum(new_count, self.cap)
#
#         # compute new mean. Note that once we reach self.cap (e.g. 1000), the 'previous mean' matters less
#         new_mean = pre_mean * (1-alpha) + (this_sum/this_bs) * alpha
#
#         updates = [(self.count, new_count), (self.mean, new_mean)]
#         self.add_update(updates, x)
#
#         # the first few 1000 should not matter that much towards this cost
#         return K.minimum(1., new_count/self.cap) * K.expand_dims(new_mean, 0)
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#

class LocalLinear(Layer):
    """ 
    Local linear layer: each pixel/voxel has its own linear operation (two parameters)
    out[v] = a * in[v] + b
    """

    def __init__(self, my_initializer='RandomNormal', **kwargs):
        self.initializer = my_initializer
        super(LocalLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.mult = self.add_weight(name='mult-kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias-kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalLinear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.mult + self.bias 

    def compute_output_shape(self, input_shape):
        return input_shape
 


