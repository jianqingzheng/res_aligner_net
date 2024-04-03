"""
losses for VoxelMorph
"""


# Third party inports
import tensorflow as tf
import keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import ReLU
import numpy as np
import sys

# from yipeng hu MedIA2019>>
def weighted_binary_cross_entropy(ts, ps, pw=1, eps=1e-6):
    ps = tf.clip_by_value(ps, eps, 1-eps)
    return -tf.reduce_sum(
        tf.concat([ts*pw, 1-ts], axis=4)*tf.log(tf.concat([ps, 1-ps], axis=4)),
        axis=4, keep_dims=True)

def dice_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts*ps, axis=[1, 2, 3, 4]) * 2
    denominator = tf.reduce_sum(ts, axis=[1, 2, 3, 4]) + tf.reduce_sum(ps, axis=[1, 2, 3, 4])+eps_vol
    return numerator/denominator


def dice_generalised(ts, ps, weights):
    ts2 = tf.concat([ts, 1-ts], axis=4)
    ps2 = tf.concat([ps, 1-ps], axis=4)
    numerator = 2 * tf.reduce_sum(tf.reduce_sum(ts2*ps2, axis=[1, 2, 3]) * weights, axis=1)
    denominator = tf.reduce_sum((tf.reduce_sum(ts2, axis=[1, 2, 3]) +
                                 tf.reduce_sum(ps2, axis=[1, 2, 3])) * weights, axis=1)
    return numerator/denominator


def jaccard_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts*ps, axis=[1, 2, 3, 4])
    denominator = tf.reduce_sum(tf.square(ts), axis=[1, 2, 3, 4]) + \
                  tf.reduce_sum(tf.square(ps), axis=[1, 2, 3, 4]) - numerator + eps_vol
    return numerator/denominator


def gauss_kernel1d(sigma):
    if sigma == 0:
        return 0
    else:
        tail = int(sigma*3)
        k = tf.exp([-0.5*x**2/sigma**2 for x in range(-tail, tail+1)])
        return k / tf.reduce_sum(k)


def cauchy_kernel1d(sigma):  # this is an approximation
    if sigma == 0:
        return 0
    else:
        tail = int(sigma*5)
        # k = tf.reciprocal(([((x/sigma)**2+1)*sigma*3.141592653589793 for x in range(-tail, tail+1)]))
        k = tf.reciprocal([((x/sigma)**2+1) for x in range(-tail, tail + 1)])
        return k / tf.reduce_sum(k)


def separable_filter3d(vol, kernel):
    if kernel == 0:
        return vol
    else:
        strides = [1, 1, 1, 1, 1]
        return tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
            vol,
            tf.reshape(kernel, [-1, 1, 1, 1, 1]), strides, "SAME"),
            tf.reshape(kernel, [1, -1, 1, 1, 1]), strides, "SAME"),
            tf.reshape(kernel, [1, 1, -1, 1, 1]), strides, "SAME")


def single_scale_loss(label_fixed, label_moving, loss_type):
    if loss_type == 'cross-entropy':
        label_loss_batch = tf.reduce_mean(weighted_binary_cross_entropy(label_fixed, label_moving), axis=[1, 2, 3, 4])
    elif loss_type == 'mean-squared':
        label_loss_batch = tf.reduce_mean(tf.squared_difference(label_fixed, label_moving), axis=[1, 2, 3, 4])
    elif loss_type == 'dice':
        label_loss_batch = 1 - dice_simple(label_fixed, label_moving)
    elif loss_type == 'jaccard':
        label_loss_batch = 1 - jaccard_simple(label_fixed, label_moving)
    else:
        raise Exception('Not recognised label correspondence loss!')
    return label_loss_batch


def multi_scale_loss(label_fixed, label_moving, loss_type, loss_scales):
    label_loss_all = tf.stack(
        [single_scale_loss(
            separable_filter3d(label_fixed, gauss_kernel1d(s)),
            separable_filter3d(label_moving, gauss_kernel1d(s)), loss_type)
            for s in loss_scales],
        axis=1)
    return tf.reduce_mean(label_loss_all, axis=1)
# edited by jianqing
class MSLE():
    """
        Multi-Scale Label Error
        """
    def __init__(self, scale_num=4,num_lab=1, loss_type='nse',scale_type='pool', eps=1e2):
        # self.win = win
        self.eps = eps
        self.scales = [2**(i) for i in range(0,scale_num)]
        self.scale_num = scale_num
        self.ndims=3
        # self.type = type
        if loss_type == 'dsc':
            self._loss=self._dsc
        elif loss_type=='nse':
            self._loss = self._nse
        if scale_type=='pool':
            self.pooling=[KL.MaxPool3D(pool_size=s) for s in self.scales]
            # self.pooling = [KL.AveragePooling3D(pool_size=s) for s in self.scales]
            self._scale_vary = self._pool
        else:
            self.eyes = tf.reshape(tf.eye(num_lab),[1]*self.ndims+[num_lab,num_lab])
            self.kernels = [self._build_kernel(s) for s in self.scales]
            self._scale_vary = self._smooth
    def _build_kernel(self,std):
        if std == 0:
            kernel = 1
        else:
            tail = int(std * 3)
            k = tf.exp([-1.5 * x ** 2 / std ** 2 for x in range(-tail, tail + 1)])
            kernel = k / tf.reduce_sum(k)
        return tf.reshape(kernel, [-1, 1, 1, 1, 1])*tf.reshape(kernel, [1, -1, 1, 1, 1])*tf.reshape(kernel, [1, 1, -1, 1, 1])*self.eyes
    def _smooth(self,vol,scale_id,strides=1):
        if self.scales[scale_id] ==0:
            return vol
        else:
            return tf.nn.conv3d(vol, self.kernels[scale_id],[strides]*(2+self.ndims), padding="SAME")
    def _pool(self,vol,scale_id):
        if self.scales[scale_id] <=1:
            return vol
        else:
            return self.pooling[scale_id](vol)

    def _nse(self, ps,ts):
        numerator = tf.reduce_sum(tf.square(ts - ps), axis=[1, 2, 3]) 
        denominator = tf.reduce_sum(ts, axis=[1, 2, 3]) + self.eps# + tf.reduce_sum(ps, axis=[1, 2, 3]) + self.eps
        return 1.*(numerator / denominator)

    def _dsc(self, ts, ps):
        numerator = tf.reduce_sum(ts * ps, axis=[1, 2, 3]) * 2
        denominator = tf.reduce_sum(ts+ps, axis=[1, 2, 3]) + self.eps# + tf.reduce_sum(ps, axis=[1, 2, 3]) + self.eps
        return 1 - (numerator / denominator)

    def loss(self, I, J, thresh=0.5):
        # I = tf.sigmoid(5 * (I - thresh))
        # J = tf.sigmoid(5 * (J - thresh))
        return tf.reduce_mean(tf.stack([self._loss(self._scale_vary(I,i),self._scale_vary(J,i)) for i in range(self.scale_num)],-1))
# from yipeng hu MedIA2019<<

# def boundary_clip(vol, volshape=volshape, lab=True):
#     ndims = 3
#     # lab=True
#     for i in range(ndims):
#         vol[..., i] = np.minimum(np.maximum(vol[..., i], 0), volshape[i] - 1)
#         lab = np.logical_and(np.logical_and(lab, vol[..., i] >= 0), vol[..., i] < volshape[i])
#     return [vol, lab]
#
# def eval_tre(vol1=None, vol2=None, disp=None,dispshape=None, kp=[None,None], crop_sz=None,reader=None):
#     ndims = 3
#     # [kp_pth1, kp_pth2] = kp_paths
#     # kp_data1 = reader(kp_pth1)
#     # kp_data2 = reader(kp_pth2)
#     # kp1 = kp_data1.to_numpy() / 2 - crop_sz
#     # kp2 = kp_data2.to_numpy() / 2 - crop_sz
#     [kp1,kp2]=kp
#     # dispshape = disp.shape[:ndims]
#     lab = True
#     [kp1, lab] = boundary_clip(kp1, dispshape, lab=lab)
#     [kp2, lab] = boundary_clip(kp2, dispshape, lab=lab)
#     # a=np.round(np.transpose(kp2)).astype(int)
#     # a=np.ravel_multi_index(np.round(np.transpose(kp2)).astype(int),dispshape)
#     # kp_tmp = kp1
#     kp_tmp = kp2
#
#     kp_ddf = np.take(np.reshape(disp, [np.prod(dispshape), 3]),np.ravel_multi_index(np.round(np.transpose(kp_tmp)).astype(int), dispshape), axis=0)
#     # kp_ddf = 0
#     # return np.mean(np.sqrt(np.sum(np.square(kp1 - kp2 + kp_ddf), axis=-1)))
#     return np.mean(np.sqrt(np.sum(np.square(kp1 - kp2 - kp_ddf), axis=-1)[lab])) / 2

def dsc(bv1,bv2,thresh=0.5,eps=10**3):
    # eps = 10 ** 3
    if thresh is not None:
        bv1 = tf.sigmoid(10*(bv1-thresh))
        bv2 = tf.sigmoid(10*(bv2-thresh))
    intersection = tf.reduce_sum(bv1*bv2,axis=[1,2,3])
    addedsection = tf.reduce_sum(bv1+bv2,axis=[1,2,3])
    return tf.reduce_mean(1-(2. * (intersection+eps) / (addedsection + eps)))


def mse(x,y,L=None,relative_axis=None):
    # return tf.reduce_mean(tf.square(x-y))
    if relative_axis is None:
        return tf.reduce_mean((tf.square(x - y))) if L is None else tf.reduce_mean(
            (tf.square(x - y) * tf.cast(L, dtype='float32')))
    else:
        eps=1e-5
        return tf.reduce_mean(tf.reduce_sum(tf.square(x - y),axis=relative_axis)/(eps+tf.reduce_sum(tf.square(y),axis=relative_axis)))

class MSE():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps


    def mse(self, I, J, L=None):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        # I = tf.print(I, 'y_true = ')
        # J = tf.print(J, 'y_pred = ')
        tf.print(I, output_stream=sys.stdout)
        tf.print(J, output_stream=sys.stdout)
        # return negative cc.
        return tf.reduce_mean((tf.square(I-J))) if L is None else tf.reduce_mean((tf.square(I-J)*tf.cast(L,dtype='float32')))

    def loss(self, I, J, L=None):
        return self.mse(I, J, L)

def binary_dice(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = 1 + np.range(ndims)

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice

class MSSL():
    """
    Multi-Scale Similarity Loss (over window)
    """

    def __init__(self,scale_num=3, win=None,std=1.5,type='ssim', eps=1e-5):
        self.win = win
        self.eps = eps
        self.std=std
        self.scale_num=scale_num
        self.type=type

    def mscv(self,I,J):
        ndims = len(I.get_shape().as_list()) - 2
        downsample_layer = getattr(KL, 'AveragePooling%dD' % ndims)
        Covars = []
        Covars.append(self.lcovar(I, J))
        for i in range(self.scale_num):
            I=downsample_layer()(I)
            J=downsample_layer()(J)
            Covars.append(self.lcovar(I, J))
        return tf.reduce_mean(Covars)

    def lcovar(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        # set window size
        if self.win is None:
            self.win = 9
        self.win = [self.win]* ndims
        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)
        # compute CC squares
        # I2 = I*I
        # J2 = J*J
        IJ = I*J
        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])/np.prod(self.win)
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_avg = conv_fn(I, sum_filt, strides, padding)
        J_avg = conv_fn(J, sum_filt, strides, padding)
        # I2_sum = conv_fn(I2, sum_filt, strides, padding)
        # J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_avg = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        # win_size = np.prod(self.win)
        # u_I = I_sum/win_size
        # u_J = J_sum/win_size

        covar = IJ_avg - I_avg*J_avg #+ u_I*u_J#*win_size

        return tf.reduce_mean(covar)

    def loss(self, I, J):
        return 1-tf.image.ssim_multiscale(I,J,1,power_factors=[1]*self.scale_num,filter_size=self.win,filter_sigma=self.std)/tf.cast(self.scale_num,dtype=tf.float32) if self.type =='ssim' else - self.mscv(I, J)
    
class MSCC():
    """
    Multi-Scale local (over window) normalized cross correlation
    """

    def __init__(self,scale_num=3, win=None, eps=1e-5):
        self.win = win
        self.eps = eps
        self.scale_num=scale_num

    def mscc(self,I,J):
        ndims = len(I.get_shape().as_list()) - 2
        downsample_layer = getattr(KL, 'AveragePooling%dD' % ndims)
        CCs = []
        CCs.append(self.ncc(I, J))
        for i in range(self.scale_num):
            I=downsample_layer()(I)
            J=downsample_layer()(J)
            CCs.append(self.ncc(I, J))
        return tf.reduce_mean(CCs)

    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return - self.mscc(I, J)

class NCC():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None,num_ch=1, eps=1e-5,central=True,smoonth=False):
        self.win = win
        self.eps = eps
        self.eyes = tf.reshape(tf.eye(num_ch), [1] * 3 + [num_ch, num_ch])
        self.central=central
        self.ndims=3
        self.strides = [1] * (self.ndims + 2)
        # set window size
        if self.win is None:
            self.win = [8] * self.ndims
        if smoonth:
            self.kernels = self._build_kernel(std=.5)
        self.smoonth = smoonth
        self.sum_filt=self._build_kernel(std=0.)
    def _build_kernel(self, std=0.):
        if std == 0.:
            return tf.ones([*self.win, 1, 1]) * self.eyes
        else:
            tail = int(np.ceil(std)) * 3
            k = tf.exp([-0.5 * x ** 2 / std ** 2 for x in range(-tail, tail + 1)])
            kernel = k / tf.reduce_sum(k)
            return tf.reshape(kernel, [-1, 1, 1, 1, 1]) * tf.reshape(kernel, [1, -1, 1, 1, 1]) * tf.reshape(kernel,[1, 1, -1, 1, 1]) * self.eyes

    def ncc(self, I, J, label=None):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        # ndims = len(I.get_shape().as_list()) - 2
        # assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        padding = 'SAME'


        # compute filters

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % self.ndims)
        if self.smoonth:
            I = conv_fn(I, self.kernels, self.strides, padding)
            J = conv_fn(J, self.kernels, self.strides, padding)
        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        if self.central:
            # compute local sums via convolution
            I_sum = conv_fn(I, self.sum_filt, self.strides, padding)
            J_sum = conv_fn(J, self.sum_filt, self.strides, padding)
            I2_sum = conv_fn(I2, self.sum_filt, self.strides, padding)
            J2_sum = conv_fn(J2, self.sum_filt, self.strides, padding)
            IJ_sum = conv_fn(IJ, self.sum_filt, self.strides, padding)
            # compute cross correlation
            win_size = np.prod(self.win)

            # u_I = I_sum/win_size
            # u_J = J_sum/win_size
            # cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
            # I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
            # J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
            cross = tf.reduce_sum(IJ_sum, axis=-1) - tf.reduce_sum(I_sum*J_sum, axis=-1)/win_size
            I_var = tf.reduce_sum(I2_sum, axis=-1) - tf.reduce_sum(I_sum*I_sum, axis=-1)/win_size
            J_var = tf.reduce_sum(J2_sum, axis=-1) - tf.reduce_sum(J_sum*J_sum, axis=-1)/win_size
        else:
            # compute local sums via convolution
            I2_sum = conv_fn(I2, self.sum_filt, self.strides, padding)
            J2_sum = conv_fn(J2, self.sum_filt, self.strides, padding)
            IJ_sum = conv_fn(IJ, self.sum_filt, self.strides, padding)

            cross = tf.reduce_sum(IJ_sum, axis=-1)
            I_var = tf.reduce_sum(I2_sum, axis=-1)
            J_var = tf.reduce_sum(J2_sum, axis=-1)

        cc = (cross*cross / (I_var*J_var + self.eps))
        if label is not None:
            label=tf.reduce_sum(tf.cast(label, dtype='float32'),axis=-1)
            cc = tf.reduce_mean(cc * label, axis=[1, 2, 3],keepdims=False) / (tf.reduce_mean(label, axis=[1, 2, 3],keepdims=False) + self.eps)
            # cc= cc
            # return negative cc.
        return tf.reduce_mean(cc)


    def loss(self, I, J, label=None):
        return - self.ncc(I, J, label=label)

class Grad():
    """
    N-D gradient loss
    """

    def __init__(self, penalty=['l1'],eps=1e-8,outrange_weight=80,apear_scale=9):
        self.penalty = penalty
        self.eps = eps
        self.outrange_weight=outrange_weight
        self.apear_scale=apear_scale
    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)
        return df

    def _eq_diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)
        pad=[[1,0]]+[[0,0]]*(ndims+1)
        df = [None] * ndims
        for i in range(ndims):
            # df_full = tf.zeros_like(y, dtype=float)
            d = i + 1
            # permute dimensions to put the ith dimension first
            # r = [d, *range(d), *range(d + 1, ndims + 2)]
            # y = K.permute_dimensions(y, r)
            yt = tf.transpose(y, [d, *range(d), *range(d + 1, ndims + 2)])
            # dfi = y[1:, ...] - y[:-1, ...]
            # df_full=tf.pad(yt[1:, ...] - yt[:-1, ...], pad)
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            # r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            # df[i] = K.permute_dimensions(df_full, r)
            df[i] = tf.transpose(tf.pad(yt[1:, ...] - yt[:-1, ...], pad), [*range(1, d + 1), 0, *range(d + 1, ndims + 2)])
        return df

    def _outl_dist(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        # ndims = len(vol_shape)
        select_loc=[s//2 for s in vol_shape]
        # disp=y[select_loc]
        act=ReLU()
        # disp=tf.gather_nd(y,)
        return tf.reduce_mean(act(tf.abs(y[:, select_loc[0],select_loc[1],select_loc[2], :]) - tf.cast(tf.expand_dims(y.get_shape()[1:-1], 0),tf.float32)))
        # return tf.reduce_mean(act(tf.abs(y[:,select_loc[:],:])-tf.expand_dims(y.get_shape()[1:-1], 0)))

    def _eval_detJ(self, disp=None, weight=None):
        ndims = 3
        # label = vol2 > thresh
        # label=vol2[...,0]>thresh
        # label=np.ones_like(vol2[...,0])
        # a=np.stack(np.gradient(disp,axis=[-2,-3,-4]),-1)
        # b=np.sum(label)
        # rescale_factor = 2
        # disp = zoom(disp, [rescale_factor] * ndims + [1], mode='nearest')
        # label=zoom(label, rescale_factor, mode='nearest')
        # a=np.stack(np.gradient(disp,axis=[-2,-3,-4]),-1)
        # b = np.linalg.det(a)
        # weight = 1 if weight is None else weight[...,0]
        weight = 1
        # tf.stack(disp, -1)
        detj = (disp[0][..., 0] * disp[1][..., 1] * disp[2][..., 2]) + (
                    disp[0][..., 1] * disp[1][..., 2] * disp[2][..., 0]) + (
                           disp[0][..., 2] * disp[1][..., 0] * disp[2][..., 1]) - (
                           disp[0][..., 2] * disp[1][..., 1] * disp[2][..., 0]) - (
                           disp[0][..., 0] * disp[1][..., 2] * disp[2][..., 1]) - (
                           disp[0][..., 1] * disp[1][..., 0] * disp[2][..., 2])
        return tf.reduce_mean(tf.nn.relu(-detj) * weight)
        # weight = 1 if weight is None else weight
        # return tf.reduce_mean(tf.nn.relu(-tf.linalg.det(tf.stack(disp, -1))) * weight)

    def loss(self, x=None, y_pred=None,img=None):
        reg_loss = 0
        if img is None:
            # if self.penalty == 'l1':
            if 'l1' in self.penalty:
                df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
                reg_loss += tf.add_n(df) / len(df)
                # return (tf.add_n(df) / len(df)) + self.outrange_weight * self._outl_dist(y_pred)
            # elif self.penalty == 'l2':
            if 'l2' in self.penalty:
                # assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
                df = [tf.reduce_mean(f * f) for f in self._diffs(y_pred)]
                reg_loss += tf.add_n(df) / len(df)
                # return (tf.add_n(df) / len(df)) + self.outrange_weight * self._outl_dist(y_pred)
        else:
            dg = tf.exp(-self.apear_scale*tf.add_n([tf.reduce_sum(g*g,axis=-1,keepdims=True) for g in self._eq_diffs(img)])/tf.reduce_sum(tf.square(.2+img),axis=-1,keepdims=True))
            if 'l1' in self.penalty:
                df = [tf.reduce_mean(tf.abs(f) * dg) for f in self._eq_diffs(y_pred)]
                reg_loss += tf.add_n(df) / len(df)
                # return (tf.add_n(df) / len(df)) + self.outrange_weight * self._outl_dist(y_pred)
            # elif self.penalty == 'l2':
            if 'l2' in self.penalty:
                # assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
                df = [tf.reduce_mean(f * f * dg) for f in self._eq_diffs(y_pred)]
                reg_loss += tf.sqrt(tf.add_n(df) / len(df))
                # return (tf.add_n(df) / len(df)) + self.outrange_weight * self._outl_dist(y_pred)
        if 'detj' in self.penalty:
            df = self._eval_detJ(self._eq_diffs(y_pred))#, dg[...,0])
            reg_loss += df#0.5*df
        return reg_loss + self.outrange_weight * self._outl_dist(y_pred)



class Miccai2018():
    """
    N-D main loss for VoxelMorph MICCAI Paper
    prior matching (KL) term + image matching term
    """

    def __init__(self, image_sigma, prior_lambda, flow_vol_shape=None):
        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.D = None
        self.flow_vol_shape = flow_vol_shape


    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently, 
        has a '1' in the immediate neighbor, and 0 elsewehre.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied 
        # ith feature to ith feature
        filt = np.zeros([3] * ndims + [ndims, ndims])
        for i in range(ndims):
            filt[..., i, i] = filt_inner
                    
        return filt


    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)
        sz = [*vol_shape, ndims]

        # prepare conv kernel
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # prepare tf filter
        z = K.ones([1] + sz)
        filt_tf = tf.convert_to_tensor(self._adj_filt(ndims), dtype=tf.float32)
        strides = [1] * (ndims + 2)
        return conv_fn(z, filt_tf, strides, "SAME")


    def prec_loss(self, y_pred):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter, 
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        vol_shape = y_pred.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)
        
        sm = 0
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y_pred, r)
            df = y[1:, ...] - y[:-1, ...]
            sm += K.mean(df * df)

        return 0.5 * sm / ndims


    def kl_loss(self, y_true, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        ndims = len(y_pred.get_shape()) - 2
        mean = y_pred[..., 0:ndims]
        log_sigma = y_pred[..., ndims:]
        if self.flow_vol_shape is None:
            # Note: this might not work in multi_gpu mode if vol_shape is not apriori passed in
            self.flow_vol_shape = y_true.get_shape().as_list()[1:-1]

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims, 
        # which is a function of the data
        if self.D is None:
            self.D = self._degree_matrix(self.flow_vol_shape)

        # sigma terms
        sigma_term = self.prior_lambda * self.D * tf.exp(log_sigma) - log_sigma
        sigma_term = K.mean(sigma_term)

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(mean)

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term) # ndims because we averaged over dimensions as well


    def recon_loss(self, y_true, y_pred):
        """ reconstruction loss """
        return 1. / (self.image_sigma**2) * K.mean(K.square(y_true - y_pred))


class SparseVM(object):
    '''
    SparseVM Sparse Normalized Local Cross Correlation (SLCC)
    '''
    def __init__(self, mask):
        self.mask = mask

    def conv_block(self,data, mask, conv_layer, mask_conv_layer, core_name):
        '''
        data is the data tensor
        mask is a binary tensor the same size as data

        steps:
        - set empty voxels in data using data *= mask
        - conv data and mask with the conv conv_layer
        - re-weight data
        - binarize mask
        '''

        # mask.dtype
        # data.dtype
        # make sure the data is sparse according to the mask
        wt_data = keras.layers.Lambda(lambda x: x[0] * x[1], name='%s_pre_wmult' % core_name)([data, mask])
        # convolve data
        conv_data = conv_layer(wt_data)  
    
        # convolve mask
        conv_mask = mask_conv_layer(mask)
        zero_mask = keras.layers.Lambda(lambda x:x*0+1)(mask)
        conv_mask_allones = mask_conv_layer(zero_mask) # all_ones mask to get the edge counts right.
        mask_conv_layer.trainable = False
        o = np.ones(mask_conv_layer.get_weights()[0].shape)
        mask_conv_layer.set_weights([o])
    
        # re-weight data (this is what makes the conv makes sense)
        data_norm = lambda x: x[0] / (x[1] + 1e-2)
        # data_norm = lambda x: x[0] / K.maximum(x[1]/x[2], 1)
        out_data = keras.layers.Lambda(data_norm, name='%s_norm_im' % core_name)([conv_data, conv_mask])
        mask_norm = lambda x: tf.cast(x > 0, tf.float32)
        out_mask = keras.layers.Lambda(mask_norm, name='%s_norm_wt' % core_name)(conv_mask)

        return (out_data, out_mask, conv_data, conv_mask)


         
    def sparse_conv_cc3D(self, atlas_mask, conv_size = 13, sum_filter = 1, padding = 'same', activation = 'elu'):
        '''
        Sparse Normalized Local Cross Correlation (SLCC) for 3D images
        '''
        def loss(I, J):
            # pass in mask to class: e.g. Mask(model.get_layer("mask").output).sparse_conv_cc3D(atlas_mask),
            mask = self.mask
            # need the next two lines to specify channel for source image (otherwise won't compile)
            I = I[:,:,:,:,0]
            I = tf.expand_dims(I, -1)
             
            I2 = I*I
            J2 = J*J
            IJ = I*J
            input_shape = I.shape
            # want the size without the channel and batch dimensions
            ndims = len(input_shape) -2
            strides = [1] * ndims
            convL = getattr(KL, 'Conv%dD' % ndims)
            im_conv = convL(sum_filter, conv_size, padding=padding, strides=strides,kernel_initializer=keras.initializers.Ones())
            im_conv.trainable = False
            mask_conv = convL(1, conv_size, padding=padding, use_bias=False, strides=strides,kernel_initializer=keras.initializers.Ones())
            mask_conv.trainable = False

            combined_mask = mask*atlas_mask
            u_I, out_mask_I, not_used, conv_mask_I = self.conv_block(I, mask, im_conv, mask_conv, 'u_I')
            u_J, out_mask_J, not_used, conv_mask_J = self.conv_block(J, atlas_mask, im_conv, mask_conv, 'u_J')
            not_used, not_used_mask, I_sum, conv_mask = self.conv_block(I, combined_mask, im_conv, mask_conv, 'I_sum')
            not_used, not_used_mask, J_sum, conv_mask = self.conv_block(J, combined_mask, im_conv, mask_conv, 'J_sum')
            not_used, not_used_mask, I2_sum, conv_mask = self.conv_block(I2, combined_mask, im_conv, mask_conv, 'I2_sum')
            not_used, not_used_mask, J2_sum, conv_mask = self.conv_block(J2, combined_mask, im_conv, mask_conv, 'J2_sum')
            not_used, not_used_mask, IJ_sum, conv_mask = self.conv_block(IJ, combined_mask, im_conv, mask_conv, 'IJ_sum')
    
            cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*conv_mask
            I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*conv_mask
            J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*conv_mask
            cc = cross*cross / (I_var*J_var + 1e-2) 
            return -1.0 * tf.reduce_mean(cc)
        return loss
