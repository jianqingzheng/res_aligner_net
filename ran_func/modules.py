"""
Networks for RAN model
"""
# main imports
from typing import Dict, List, Optional, Tuple, Union
# third party
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, concatenate

# import tensorflow.python.keras.backend as K
# from tensorflow.python.keras.models import Model
# import tensorflow.python.keras.layers as KL
# from tensorflow.python.keras.layers import Layer
# from tensorflow.python.keras.layers import Input, concatenate

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# tf.config.run_functions_eagerly(True)

from external.neuron import layers as nrn_layers
from external.neuron import utils as nrn_utils

########################################################################################################

eps=1e-6

def rigid_transform_3D(A, B):

    # find mean column wise
    centroid_A = tf.reduce_mean(A, axis=1)
    centroid_B = tf.reduce_mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = tf.reshape(centroid_A,[-1, 1])
    centroid_B = tf.reshape(centroid_B,[-1, 1])

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # H = Am @ tf.transpose(Bm)
    H = tf.matmul(Am, Bm, transpose_a=False, transpose_b=True)

    # find rotation
    U, S, V = tf.linalg.svd(H)
    # R = Vt.T @ U.T
    R = tf.matmul(V, U, transpose_a=False, transpose_b=True)
    # special reflection case
    if tf.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        # V[2,:] *= -1
        V=V[..., 2, :].assign(V[..., 2, :]*-1)
        # R = V.T @ U.T
        R = tf.matmul(V, U, transpose_a=False, transpose_b=True)

    # t = -R @ centroid_A + centroid_B
    t = centroid_B - tf.matmul(R, centroid_A, transpose_a=False, transpose_b=False)
    return R, t


def ddf_decouple(ddf_vol,masks,ndims=3):
    size_vol=masks.shape
    ddf_rigid = np.zeros_like(ddf_vol)
    for b in range(size_vol[0]):
        for m in range(size_vol[-1]):
            _,_,ddf_r=ddf2param(ddf_vol[b,...],masks[b,...,m],ndims=ndims)
            ddf_rigid[b,...]+=ddf_r
    ddf_deform=ddf_vol-ddf_rigid
    return ddf_rigid,ddf_deform

def ddf2param(ddf_vol,mask,ndims=3,sample_rate=7):
    orig_shape=ddf_vol.shape
    ddf_reshape=np.reshape(ddf_vol,[-1,orig_shape[-1]])
    mask_reshape=np.reshape(mask,[-1,1])
    index=np.nonzero(mask)

    # index=tuple([idx[:sample_rate:] for idx in index])
    index_flat=np.nonzero(mask_reshape)
    coord_src=np.stack(index,axis=0)

    # b=(*index,np.ones_like(index[0])*0)
    # a=[ddf_vol[(*index,np.ones_like(index[0])*d)] for d in range(orig_shape[-1])]
    coord_ddf=np.stack([ddf_vol[(*index,np.ones_like(index[0])*d)] for d in range(orig_shape[-1])],axis=0)
    coord_tgt=coord_ddf+coord_src
    R, t = rigid_transform_3D(coord_src[...,::sample_rate], coord_tgt[...,::sample_rate])
    # coord_rig= np.transpose((R@np.transpose(coord_src)) + t)
    coord_rig = (R @ coord_src) + t
    ddf_def = np.zeros_like(ddf_vol)
    ddf_rig = np.zeros_like(ddf_vol)
    for d in range(orig_shape[-1]):
        ddf_def[(*index, np.ones_like(index[0]) * d)] = coord_tgt[d, ...] - coord_rig[d, ...]
        ddf_rig[(*index, np.ones_like(index[0]) * d)] = coord_rig[d, ...] - coord_src[d, ...]

    param=np.concatenate([R,t],axis=1)
    return param,ddf_def,ddf_rig
########################################################################################################
class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 momentum=0.9,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = KL.InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=True):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': tf.compat.v1.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.compat.v1.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tf.compat.v1.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.compat.v1.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': tf.compat.v1.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.compat.v1.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Resize3d(KL.Layer):
    """
    Resize image in two folds.

    - resize dim2 and dim3
    - resize dim1 and dim2
    """

    def __init__(
        self,
        method: str = tf.image.ResizeMethod.BILINEAR,
        name: str = "resize3d",
        scale: Optional[tuple] = (2,2,2),
        shape: Optional[tuple] = (64,64,64),
    ):
        """
        Init, save arguments.

        :param shape: (dim1, dim2, dim3)
        :param method: tf.image.ResizeMethod
        :param name: name of the layer
        """
        super().__init__(name=name)
        assert len(shape) == 3
        assert shape is not None or scale is not None
        self._shape = shape
        self._method = method
        self._scale = scale

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Perform two fold resize.

        :param inputs: shape = (batch, dim1, dim2, dim3, channels)
                                     or (batch, dim1, dim2, dim3)
                                     or (dim1, dim2, dim3)
        :param kwargs: additional arguments
        :return: shape = (batch, out_dim1, out_dim2, out_dim3, channels)
                                or (batch, dim1, dim2, dim3)
                                or (dim1, dim2, dim3)
        """
        # sanity check
        image = inputs
        # self._shape=[shape//scale for shape,scale in zip(image.shape,self._scale)]
        image_dim = len(image.shape)

        # init
        if image_dim == 5:
            has_channel = True
            has_batch = True
            input_image_shape = image.shape[1:4]
        elif image_dim == 4:
            has_channel = False
            has_batch = True
            input_image_shape = image.shape[1:4]
        elif image_dim == 3:
            has_channel = False
            has_batch = False
            input_image_shape = image.shape[0:3]
        else:
            raise ValueError(
                "Resize3d takes input image of dimension 3 or 4 or 5, "
                "corresponding to (dim1, dim2, dim3) "
                "or (batch, dim1, dim2, dim3) "
                "or (batch, dim1, dim2, dim3, channels), "
                "got image shape{}".format(image.shape)
            )

        # no need of resize
        if input_image_shape == tuple(self._shape):
            return image

        # expand to five dimensions
        if not has_batch:
            image = tf.expand_dims(image, axis=0)
        if not has_channel:
            image = tf.expand_dims(image, axis=-1)
        assert len(image.shape) == 5  # (batch, dim1, dim2, dim3, channels)
        image_shape = tf.shape(image)

        # merge axis 0 and 1
        output = tf.reshape(
            image, (-1, image_shape[2], image_shape[3], image_shape[4])
        )  # (batch * dim1, dim2, dim3, channels)

        # resize dim2 and dim3
        output = tf.image.resize(
            images=output, size=self._shape[1:3], method=self._method
        )  # (batch * dim1, out_dim2, out_dim3, channels)

        # split axis 0 and merge axis 3 and 4
        output = tf.reshape(
            output,
            shape=(-1, image_shape[1], self._shape[1], self._shape[2] * image_shape[4]),
        )  # (batch, dim1, out_dim2, out_dim3 * channels)

        # resize dim1 and dim2
        # print(self._shape)
        output = tf.image.resize(
        # output = tf.image.resize_images(
            images=output, size=tf.constant(self._shape[:2]), method=self._method
        )  # (batch, out_dim1, out_dim2, out_dim3 * channels)

        # reshape
        output = tf.reshape(
            output, shape=[-1, *self._shape, image_shape[4]]
        )  # (batch, out_dim1, out_dim2, out_dim3, channels)

        # squeeze to original dimension
        if not has_batch:
            output = tf.squeeze(output, axis=0)
        if not has_channel:
            output = tf.squeeze(output, axis=-1)
        return output

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["shape"] = self._shape
        config["method"] = self._method
        return config


def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    transform an affine matrix to a dense location shift tensor in tensorflow

    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid

    Parameters:
        affine_matrix: ND+1 x ND+1 or ND x ND+1 matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)

    Returns:
        shift field (Tensor) of size *volshape x N

    TODO: 
        allow affine_matrix to be a vector of size nb_dims * (nb_dims + 1)
    """

    if isinstance(volshape, (tf.Dimension, tf.TensorShape)):
        volshape = volshape.as_list()

    if affine_matrix.dtype != 'float32':
        affine_matrix = tf.cast(affine_matrix, 'float32')

    nb_dims = len(volshape)
    batch_size=K.shape(affine_matrix)[0]
    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)):
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1).'
                             'Got len %d' % len(affine_matrix))

        affine_matrix = tf.reshape(affine_matrix, [nb_dims+1, nb_dims])

    if not (affine_matrix.shape[-1] in [nb_dims, nb_dims + 1] and affine_matrix.shape[-2] == (nb_dims + 1)):
        raise Exception('Affine matrix shape should match'
                        '%d+1 x %d+1 or ' % (nb_dims, nb_dims) + \
                        '%d x %d+1.' % (nb_dims, nb_dims) + \
                        'Got: ' + str(volshape))

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    # mesh = volshape2meshgrid(volshape, indexing=indexing)
    
    mesh = tf.cast(tf.tile(
        tf.expand_dims(tf.stack(volshape2meshgrid(volshape,homo=True,unit_norm=False, indexing='ij'), axis=-1), axis=0),
        [batch_size] + [1 for i in range(nb_dims + 1)]), dtype=tf.float32)
    mesh = tf.cast(mesh, 'float32')
    # if shift_center:
    #     mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]

    # add an all-ones entry and transform into a large matrix
    # flat_mesh = [flatten(f) for f in mesh]
    # flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    # mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels
    mesh_flat=tf.reshape(mesh,[batch_size,-1,4])
    # compute locations
    loc_matrix = tf.matmul(mesh_flat,affine_matrix)  # N+1 x nb_voxels
    # loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, [batch_size] + list(volshape) + [nb_dims])  # *volshape x N
    # loc = [loc[..., f] for f in range(nb_dims)]  # N-long list, each entry of shape volshape

    # get shifts and return
    return loc - mesh[...,:-1]

def volshape2meshgrid(volshape,homo=False,unit_norm=True,centralize=True, **kwargs):
    if centralize:
        if unit_norm:
            linvec = [tf.range(-1., 1. + (1. / (d - 1)), 2. / (d - 1), dtype='float32') for d in volshape]
        else:
            linvec = [tf.range(-d // 2, d // 2, dtype='float32') for d in volshape]
    else:
        if unit_norm:
            linvec = [tf.range(0., 1. + (1. / (d - 1)), 1. / (d - 1), dtype='float32') for d in volshape]
        else:
            linvec = [tf.range(0, d) for d in volshape]
    # if homo:
    #      x= nrn_utils.meshgrid(*linvec, **kwargs)+[tf.ones(volshape, dtype='float32')]
    # else:
    #     x= nrn_utils.meshgrid(*linvec, **kwargs)
    # return [tf. constant(y) for y in x]
    if homo:
        return nrn_utils.meshgrid(*linvec, **kwargs)+[tf.ones(volshape, dtype='float32')]
    else:
        return nrn_utils.meshgrid(*linvec, **kwargs)


scale_num=10
def _eq_diffs(y,ndims=3):
    pad = [[1, 0]] + [[0, 0]] * (ndims + 1)
    df = [None] * ndims
    for i in range(ndims):
        d = i + 1
        yt = tf.transpose(y, [d, *range(d), *range(d + 1, ndims + 2)])
        df[i] = tf.transpose(tf.pad(yt[1:, ...] - yt[:-1, ...], pad), [*range(1, d + 1), 0, *range(d + 1, ndims + 2)])
    return df

def ddf_filter(ddf,vol,interp_method='linear'):
    if interp_method=='edge_aware':
        loc0 = tf.floor(ddf)
        # loc1 = ddf+1
        diff_loc0 = ddf-loc0
        diff_loc1 = 1-diff_loc0
        diff_vol = tf.abs(tf.concat(_eq_diffs(vol),axis=-1))

        diff_exp_loc0 = tf.exp(scale_num * diff_vol * diff_loc0) - 1
        diff_exp_loc1 = tf.exp(scale_num * diff_vol * diff_loc1) - 1
        exp_sum = diff_exp_loc0 + diff_exp_loc1 + 1

        filt_ddf = loc0 + (diff_loc0+diff_exp_loc0)/exp_sum

        return filt_ddf
    elif interp_method=='weighted_adaptive':
        loc0 = tf.floor(ddf)
        # loc1 = ddf+1
        diff_loc0 = ddf - loc0
        diff_loc1 = 1 - diff_loc0
        diff_vol = tf.abs(vol)

        diff_exp_loc0 = tf.exp(scale_num * diff_vol * diff_loc0) - 1
        diff_exp_loc1 = tf.exp(scale_num * diff_vol * diff_loc1) - 1
        exp_sum = diff_exp_loc0 + diff_exp_loc1 + 1

        filt_ddf = loc0 + (diff_loc0 + diff_exp_loc0) / exp_sum

        return filt_ddf
    else:
        return ddf

def STN(vol_size=None,disp_size=None,name='image_warping',padding=1,vol=None,disp=None,aff_transf=None,use_aff=False,vol_feats=1,upsample_sz=1,interp_method = 'linear'):
    # inputs
    ndims = len(vol_size)
    disp_feats = ndims
    disp_size= vol_size if disp_size is None else disp_size
    if vol is None:
        vol = Input(shape=[*vol_size, vol_feats])
    if disp is None:
        disp = Input(shape=[*disp_size, disp_feats])
        model_inputs=[vol, disp]
    else:
        model_inputs=[vol]

    if upsample_sz is not 1:
        upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)(size=upsample_sz,interp_method='nearest')
        # upsample_layer = Resize3d(scale=upsample_sz)
        # disp=upsample_layer(disp)
        disp = upsample_layer(disp * upsample_sz)
    if use_aff:
        aff_transf= Input(shape=[ndims+1,ndims]) if aff_transf is None else aff_transf
        disp_aff = affine_to_shift(aff_transf, disp_size, shift_center=True)
        disp+=disp_aff

    if padding > 0:
        pad_layer = getattr(KL, 'ZeroPadding%dD' % ndims)(padding=padding)
        crop_layer = getattr(KL, 'Cropping%dD' % ndims)(cropping=padding,name='cropped'+name)
        volume = pad_layer(vol)
        displace = pad_layer(disp)
    else:
        volume=vol
        displace=disp

    spatial_transformer=nrn_layers.SpatialTransformer(name=name,interp_method=interp_method)
    interp_vol = spatial_transformer([volume, displace])
    if padding > 0:
        interp_vol = crop_layer(interp_vol)
    if use_aff:
        return Model(inputs=[vol, disp,aff_transf], outputs=[interp_vol])
    else:
        return Model(inputs=model_inputs, outputs=[interp_vol])
