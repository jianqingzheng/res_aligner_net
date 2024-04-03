"""
Networks for RAN model
"""
from tensorflow.keras.layers import ReLU, LeakyReLU

from ran_func.modules import *

########################################################################################################

eps=1e-6

def get_net(model_name=None):
    if model_name == 'RAN4':
        net_core = ran4_core
    else:
        net_core=None
    return net_core

########################################################################################################

def ran4_core(vol_size, enc_nf=None, dec_nf=None, src=None, tgt=None, src_feats=1, tgt_feats=1,num_nl=3):
    """
    RAN4+ architecture for Residual Aligner Network models with q=4 presented in the Medical Image Analysis 2024 paper.
    You may need to modify this code (e.g., vol_size) to suit your project needs.
    :param vol_size: volume size. e.g. (128, 128, 128)
    :param src_feats: number of filters for source images.
    :param tgt_feats: number of filters for target images.
    :return: the keras model
    """
    enc_nf = [8, 8, 16, 32]  # RAN4 channel num
    dec_nf = [32, 32, 16, 8, 8]
    ndims = len(vol_size)
    num_ddf=6
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    downsample_layer = getattr(KL, 'AveragePooling%dD' % ndims)
    upsample_layer0 = getattr(KL, 'UpSampling%dD' % ndims)
    upsample_layer = getattr(KL, 'Conv%dDTranspose' % ndims)
    # inputs
    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = [src, tgt]

    vSize = vol_size
    output_paddings = []
    # down-sample path (encoder)
    X=x_in
    x_enc1,x_enc2=[],[]
    stid=1
    skip_ch_num=16
    accu_padding=np.array([0]*ndims)
    for i in range(len(enc_nf)):
        X=siam_res_block(X, enc_nf[i], down_sample=downsample_layer)
        x=siam_conv_block(X,skip_ch_num)
        if i>0:
            output_paddings.append(None)
            accu_padding=accu_padding+np.array([s % 2 for s in vSize])*(2**(i-1))
            vSize = [s // 2 for s in vSize]
            x1 = tf.pad(upsample_layer0(size=2 ** (i))(x[0]), [[0, 0]] + [[0, pad] for pad in accu_padding] + [[0, 0]],mode='CONSTANT')
            x2 = tf.pad(upsample_layer0(size=2 ** (i))(x[1]), [[0, 0]] + [[0, pad] for pad in accu_padding] + [[0, 0]],mode='CONSTANT')
        else:
            output_paddings.append([s % 2 for s in vSize])
            vSize = [s // 2 for s in vSize]
            [x1,x2]=x
        x_enc1.append(x1)
        x_enc2.append(x2)

        print(X[0].shape, x_enc1[i].shape)
    Conv_nd = getattr(KL, 'Conv%dD' % ndims)
    inter_ch_num = 3 ** ndims - ndims ** 2
    ws_conv=[
        Conv_nd(3 ** ndims, kernel_size=3, padding='same', kernel_initializer='he_normal'),
        Conv_nd(num_ddf * ndims, kernel_size=1, padding='same', kernel_initializer='he_normal'),
        Conv_nd(num_ddf, kernel_size=1, padding='same', kernel_initializer='he_normal'),
        # Conv_nd(1, kernel_size=1, padding='same', kernel_initializer='he_normal'),
        Conv_nd(1 * inter_ch_num, kernel_size=3, padding='same', kernel_initializer='he_normal'),
        Conv_nd(1 * num_ddf, kernel_size=1, padding='same', kernel_initializer='he_normal'),
        # Conv_nd(inter_ch_num // 2, kernel_size=3, padding='same', kernel_initializer='he_normal'),
        Conv_nd(num_ddf, kernel_size=3, padding='same', kernel_initializer='he_normal'),
        # Conv_nd(inter_ch_num, kernel_size=3, padding='same', kernel_initializer='he_normal'),
    ]
    if num_ddf == 1:
        ws_conv.append(Conv_nd(ndims, kernel_size=3, padding='same', kernel_initializer='he_normal'))
    else:
        ws_conv.append(Conv_nd(1, kernel_size=3, padding='same', kernel_initializer='he_normal'))

    X, M, C = ws_res_align_block(siam_res_block([concatenate(x_enc1[2:]),concatenate(x_enc2[2:])], enc_nf[-1], up_sample=None), up_sample=None, M=None,C=None,ws_conv=ws_conv, output_padding=None, name='raa' + str(-1),num_ddf=num_ddf,atrous_rate=2**(i))

    ################
    # resize_layer=Resize3d(method='nearest',shape=tuple(vol_size))
    # print(tuple(vol_size))
    # mask= resize_layer(tf.nn.softmax(C[stid],axis=-1))     ###### mh-mask
    # up-sample path (decoder)

    for i in range(0,len(enc_nf)):
        X, M, C = ws_res_align_block(siam_res_block(X + [x_enc1[-i - 1],x_enc2[-i - 1]], dec_nf[i],up_sample=upsample_layer(dec_nf[i], kernel_size=2, padding='valid', strides=2,output_padding=output_paddings[-i - 1]) if len(enc_nf)-i-2<0 else None),up_sample=None, M=M, C=C,ws_conv=ws_conv, output_padding=output_paddings[-i-1],name='raa' + str(i), num_ddf=num_ddf,atrous_rate=2**(len(enc_nf)-i-2) if len(enc_nf)-i-2>=0 else 1)

    x = concatenate([X[stid], M[stid]])
    return Model(inputs=[src, tgt], outputs=[x])


def siam_conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    [x_in1, x_in2] = x_in
    ndims = len(x_in1.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)(nf, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=strides)
    x_out1 = Conv(x_in1)
    x_out1 = LeakyReLU(0.2)(x_out1)
    x_out2 = Conv(x_in2)
    x_out2 = LeakyReLU(0.2)(x_out2)

    return [x_out1,x_out2]


def siam_res_block(x_in, nf, down_sample=None,up_sample=None,dilations=[1,1,3],weight_sharing=True):
    # def siam_res_block(x_in, nf, down_sample=None,up_sample=None,dilations=[1,1,3]):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    strides=[1,1]
    if len(x_in)>2:
        [x_in1,x_in2,x_skip1,x_skip2]=x_in
        x_in1 = tf.concat([x_in1, x_skip1],-1)
        x_in2 = tf.concat([x_in2, x_skip2],-1)
    else:
        [x_in1, x_in2]=x_in
    ndims = len(x_in1.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    Conv_nd = getattr(KL, 'Conv%dD' % ndims)
    Conv0 = Conv_nd(nf, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides[0],dilation_rate=dilations[0])
    Conv1 = Conv_nd(nf, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides[0],dilation_rate=dilations[1])
    Conv2 = Conv_nd(nf, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides[0],dilation_rate=dilations[2])
    act=LeakyReLU(0.2)
    # NormLayer=KL.BatchNormalization(momentum=0.99,epsilon=0.001,center=True,scale=True)
    NormLayer1=InstanceNormalization(momentum=0.99,epsilon=0.001,center=True,scale=True)
    NormLayer2=InstanceNormalization(momentum=0.99,epsilon=0.001,center=True,scale=True)
    NormLayer3=InstanceNormalization(momentum=0.99,epsilon=0.001,center=True,scale=True)
    # branch 1
    x_in1 = Conv0(x_in1) if up_sample is None else Conv0(up_sample(x_in1))
    x_in1 = NormLayer1(x_in1)
    x_in1 = act(x_in1)
    x_out1 = Conv1(x_in1)
    x_out1 = NormLayer2(x_out1)
    x_out1 = act(x_out1)
    x_out1 = Conv2(x_out1)
    x_out1 = NormLayer3(x_out1)
    x_out1 = act(x_out1 + x_in1) if down_sample is None else down_sample()(act(x_out1 + x_in1))

    # branch 2
    if not weight_sharing:
        Conv0 = Conv_nd(nf, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides[0],
                        dilation_rate=dilations[0])
        Conv1 = Conv_nd(nf, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides[0],
                        dilation_rate=dilations[1])
        Conv2 = Conv_nd(nf, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides[0],
                        dilation_rate=dilations[2])
        # act = LeakyReLU(0.2)
        # NormLayer=KL.BatchNormalization(momentum=0.99,epsilon=0.001,center=True,scale=True)
        NormLayer1 = InstanceNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)
        NormLayer2 = InstanceNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)
        NormLayer3 = InstanceNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)
    x_in2 = Conv0(x_in2) if up_sample is None else Conv0(up_sample(x_in2))
    x_in2 = NormLayer1(x_in2)
    x_in2 = act(x_in2)
    x_out2 = Conv1(x_in2)
    x_out2 = NormLayer2(x_out2)
    x_out2 = act(x_out2)
    x_out2 = Conv2(x_out2)
    x_out2 = NormLayer3(x_out2)
    x_out2 = act(x_out2 + x_in2) if down_sample is None else down_sample()(act(x_out2 + x_in2))

    return [x_out1,x_out2]


def ws_res_align_block(x_in, down_sample=None, up_sample=None, nl=True, nf=None, M=None, C=None,ws_conv=[], output_padding=None,num_ddf=1,
                    fuse_feature_map=True, atrous_rate=1, name='raa'):
    if nl:
        leaky_ReLU=LeakyReLU(0.01)
        def boundary_limit(sample_coords0, input_size0, plus=0., minus=1.):
            sample_coords = tf.unstack(sample_coords0, axis=-1)
            # return tf.stack([tf.maximum(tf.minimum(x, sz - minus + plus), 0 + plus) for x, sz in zip(sample_coords, input_size0)],-1)
            return tf.stack([tf.maximum(tf.minimum(x, 2*sz - minus + plus), minus - 2*sz + plus) for x, sz in
                             zip(sample_coords, input_size0)], -1)

        ################
        [x_in1, x_in2] = x_in
        in_shape = x_in1.get_shape().as_list()
        batch_size = K.shape(x_in1)[0]
        nf = in_shape[-1] if nf is None else nf
        ndims = len(in_shape) - 2
        assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
        inter_ch_num=3**ndims-ndims**2
        Conv_nd = getattr(KL, 'Conv%dD' % ndims)
        Conv_val = Conv_nd(nf, kernel_size=1, padding='same', kernel_initializer='he_normal')
        Conv_resample = Conv_nd(nf, kernel_size=1, padding='same', kernel_initializer='he_normal')
        Conv_m1 = Conv_nd(inter_ch_num, kernel_size=3, dilation_rate = atrous_rate, padding='same', kernel_initializer='he_normal')
        Conv_m2 = Conv_nd(inter_ch_num, kernel_size=3, dilation_rate = atrous_rate,padding='same', kernel_initializer='he_normal')
        Conv_m3 = Conv_nd(inter_ch_num, kernel_size=3, padding='same', kernel_initializer='he_normal')
        Conv_fc = Conv_nd(inter_ch_num // 1, kernel_size=3, padding='same', kernel_initializer='he_normal')
        Conv_f = Conv_nd(inter_ch_num // 1, kernel_size=3, padding='same', kernel_initializer='he_normal')
        Conv_c = Conv_nd(1, kernel_size=1, padding='same', kernel_initializer='he_normal')
        if len(ws_conv)==0:
            Conv_m = Conv_nd(3 ** ndims, kernel_size=3, padding='same', kernel_initializer='he_normal')
            Conv_mo = Conv_nd(num_ddf * ndims, kernel_size=1, padding='same', kernel_initializer='he_normal')
            Conv_h = Conv_nd(num_ddf, kernel_size=1, padding='same', kernel_initializer='he_normal')
            Conv_c1 = Conv_nd(1 * inter_ch_num, kernel_size=3, padding='same', kernel_initializer='he_normal')
            Conv_c2 = Conv_nd(1 * num_ddf, kernel_size=1, padding='same', kernel_initializer='he_normal')
            Conv_fco = Conv_nd(num_ddf, kernel_size=3, padding='same', kernel_initializer='he_normal')

            if num_ddf == 1:
                Conv_fo = Conv_nd(ndims, kernel_size=3, padding='same', kernel_initializer='he_normal')
            else:
                Conv_fo = Conv_nd(1, kernel_size=3, padding='same', kernel_initializer='he_normal')
        else:
            [Conv_m,Conv_mo,Conv_h,Conv_c1,Conv_c2,Conv_fco,Conv_fo]=ws_conv

        NormLayer = InstanceNormalization(epsilon=0.001, center=True, scale=True,trainable=True)
        NormLayer_resamp = InstanceNormalization(epsilon=0.001, center=True, scale=True,trainable=True)
        NormLayer_conf = InstanceNormalization(epsilon=0.001, center=True, scale=True,trainable=True)
        NormLayer_fuse = InstanceNormalization(epsilon=0.001, center=True, scale=True, trainable=True)
        NormLayer_feat = InstanceNormalization(epsilon=0.001, center=True, scale=True,trainable=True)
        pad_layer = getattr(KL, 'ZeroPadding%dD' % ndims)
        crop_layer = getattr(KL, 'Cropping%dD' % ndims)

        stn = nrn_layers.SpatialTransformer()  # (name='resample_layer_'+name)

        def spatial_transformer(input, padding=1, ndims=ndims, stn=stn):
            [vol, disp] = input
            if padding > 0 and len(disp.get_shape().as_list())<ndims+3:
                vol = pad_layer(padding=padding)(vol)
                disp = pad_layer(padding=padding)(disp)
            interp_vol = stn([vol, disp])
            return crop_layer(cropping=padding)(interp_vol) if padding > 0 else interp_vol

        def correspond(xt, xs):
            mat=leaky_ReLU(Conv_m1(tf.concat([xt,xs],-1)))
            mat=leaky_ReLU(Conv_m3(leaky_ReLU(Conv_m2(mat)))+mat)
            return Conv_mo(leaky_ReLU(Conv_m(mat))),Conv_h(leaky_ReLU(NormLayer_conf(Conv_c1(mat)))),leaky_ReLU(NormLayer_feat(Conv_c2(mat)))

        def expand_prod_inv(tensor1, tensor2, act="softmax",tensor_shape=[batch_size]+in_shape[1:ndims+1]+[num_ddf*ndims]):
            if act is None:
                return tf.reshape(tf.expand_dims(tensor1, ndims+1) * tf.expand_dims(tensor2, ndims+2), tensor_shape)
            elif act is "relu":
                return tf.reshape(tf.expand_dims(tensor1, ndims+1) * tf.expand_dims(ReLU()(tensor2), ndims+2), tensor_shape)
            elif act is "softmax":
                return tf.reshape(tf.expand_dims(tensor1, ndims+1) * tf.expand_dims(tf.nn.softmax(tensor2, axis=-1), ndims+2), tensor_shape)

        def expand_prod(tensor1, tensor2,tensor3=None, act="scale+sm", tensor_shape=[batch_size*ndims] + in_shape[1:ndims + 1] + [num_ddf+1]):

            if num_ddf == 1:
                tensor2 = ReLU()(tensor2)
                return tf.concat([tensor1,tensor1*tensor2],axis=-1)
                # return tensor1*tensor2
            else:
                if act is "relu":
                    tensor2 = ReLU()(tensor2)
                elif act is "softmax":
                    tensor2 = tf.nn.softmax(tensor2, axis=-1)
                elif act is "scale+sm":
                    if tensor3 is None:
                        tensor2 = tf.nn.softmax(tensor2, axis=-1)*ReLU()(Conv_c(leaky_ReLU(tensor2)))
                    else:
                        tensor2 = tf.nn.softmax(tensor2, axis=-1) * ReLU()(Conv_c(leaky_ReLU(tensor3)))
                # tensor1=tf.expand_dims(tensor1, ndims + 1)
                # return tf.reshape(tf.transpose(tf.concat([tensor1 * tf.expand_dims(tensor2, ndims + 2),tensor1], ndims + 1),[0,ndims + 2,1,2,3,ndims + 1]), tensor_shape)
                tensor1 = tf.reshape(tensor1,[batch_size] + in_shape[1:ndims + 1] + [-1]+[ndims])
                return tf.reshape(tf.transpose(tf.concat([tensor1 * tf.expand_dims(tensor2, ndims + 2),tf.reduce_mean(tensor1,ndims+1,keepdims=True)], ndims + 1),[0,ndims + 2,1,2,3,ndims + 1]), tensor_shape)

        def unexpand(tensor,tensor_shape=[batch_size] + [ndims] + in_shape[1:ndims + 1] ):
            if num_ddf == 1:
                return tensor
            else:
                return tf.transpose(tf.reshape(tensor, tensor_shape),[0, 2, 3, 4, 1])

        if M is not None:
            if output_padding is None:
                Mu12 = M[0]
                Mu21 = M[1]
                Cu12 = C[0]
                Cu21 = C[1]
            else:
                up_pad = [[0, 0]] + [[0, pad] for pad in output_padding] + [[0, 0]]
                # upsample_layer = up_sample
                upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)(size=2) if up_sample is None else up_sample
                Mu12 = tf.pad(upsample_layer(M[0]) * 2., up_pad, mode="REFLECT")
                Mu21 = tf.pad(upsample_layer(M[1]) * 2., up_pad, mode="REFLECT")
                Cu12 = tf.pad(upsample_layer(C[0]), up_pad, mode="REFLECT")
                Cu21 = tf.pad(upsample_layer(C[1]), up_pad, mode="REFLECT")


            # if num_ddf==1:
            resampled_xin1 = (spatial_transformer([Conv_resample(x_in1), Mu21]))
            resampled_xin2 = (spatial_transformer([Conv_resample(x_in2), Mu12]))

        else:
            resampled_xin1 = Conv_resample(x_in1)
            resampled_xin2 = Conv_resample(x_in2)

        x1_phi = NormLayer(x_in1)
        x2_phi = NormLayer(x_in2)
        resampled_xin1 = NormLayer_resamp(resampled_xin1)
        resampled_xin2 = NormLayer_resamp(resampled_xin2)

        M12, C12l,C12f = correspond(x1_phi, resampled_xin2)
        M21, C21l,C21f = correspond(x2_phi, resampled_xin1)

        if M is not None:
            C12 = Conv_fco(leaky_ReLU(NormLayer_fuse(Conv_fc(tf.concat([C12l, Cu12], axis=-1)))))
            C21 = Conv_fco(leaky_ReLU(NormLayer_fuse(Conv_fc(tf.concat([C21l, Cu21], axis=-1)))))
            M12c = expand_prod(M12, (C12), C12l)
            M21c = expand_prod(M21, (C21), C21l)
            Mu12c = expand_prod(Mu12, (C12), Cu12)
            Mu21c = expand_prod(Mu21, (C21), Cu21)
            M12t = unexpand(Conv_fo(leaky_ReLU(Conv_f(tf.concat([M12c, Mu12c], axis=-1)))))
            M21t = unexpand(Conv_fo(leaky_ReLU(Conv_f(tf.concat([M21c, Mu21c], axis=-1)))))

        else:
            C12 = Conv_fco(leaky_ReLU(NormLayer_fuse(Conv_fc(C12l))))
            C21 = Conv_fco(leaky_ReLU(NormLayer_fuse(Conv_fc(C21l))))
            M12c = expand_prod(M12, (C12), C12l)
            M21c = expand_prod(M21, (C21), C21l)
            M12t = unexpand(Conv_fo(leaky_ReLU(Conv_f(M12c))))
            M21t = unexpand(Conv_fo(leaky_ReLU(Conv_f(M21c))))

        M12=M12t
        M21=M21t

        M12 = boundary_limit(M12, in_shape[1:ndims + 1])
        M21 = boundary_limit(M21, in_shape[1:ndims + 1])

        if fuse_feature_map:
            y12 = leaky_ReLU(Conv_val(tf.concat([spatial_transformer([(x_in2), M12]),C12f],axis=-1)))
            y21 = leaky_ReLU(Conv_val(tf.concat([spatial_transformer([(x_in1), M21]),C21f],axis=-1)))

            x_out1 = x_in1 + y12
            x_out2 = x_in2 + y21
        else:
            x_out1 = x_in1
            x_out2 = x_in2
        return [x_out1, x_out2], [M12, M21], [C12, C21]
    else:
        return x_in
