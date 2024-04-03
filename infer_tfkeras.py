import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob

import time
from utils import *
import ran_func as func


##############################################################################################

eps = 1e-5
np.random.seed(0)

def infer(net_core=None, model_path=None, crop_sz=None, rescale_factor=None,rescale_factor_label = 1.0,
          pair_type=None, use_lab=False, test_path=None, batch_size=1,model_name=None,int_range=None):

    suffix='*.nii.gz'
    # suffix = '*.nii'
    process_suffix = '_x2'

    if pair_type == 'paired':
        fx_pth_train = glob.glob(os.path.join(test_path, 'fixed_images'+process_suffix, ))
        mv_pth_train = glob.glob(os.path.join(test_path, 'moving_images'+process_suffix, suffix))
        fx_lab_pth = glob.glob(os.path.join(test_path, 'fixed_labels'+process_suffix, suffix))
        mv_lab_pth = glob.glob(os.path.join(test_path, 'moving_labels'+process_suffix, suffix))
        test_paths = [mv_pth_train, fx_pth_train]
        label_paths = [mv_lab_pth, fx_lab_pth]
        # syn_lab_paths = fx_lab_pth + mv_lab_pth
        volshape = np.add(nib.load(fx_pth_train[0]).dataobj.shape * np.array(rescale_factor), -2 * crop_sz).astype(int)
    else:
        test_paths = glob.glob(os.path.join(test_path, 'images'+process_suffix, suffix))
        label_paths = glob.glob(os.path.join(test_path, 'labels'+process_suffix, suffix))
        # syn_lab_paths= label_paths
        volshape = np.add(nib.load(test_paths[0]).dataobj.shape * np.array(rescale_factor), -2 * crop_sz).astype(int)
    warped_img_dir = os.path.join(test_path, 'warped_img')
    # warped_lab_dir = os.path.join(test_path, 'warped_lab')
    model_pth_name=os.path.basename(os.path.dirname(model_path))
    eval_result_dir = os.path.join(test_path, 'eval_result',model_pth_name+'.xlsx')
    print(eval_result_dir)
    tmpdir=os.path.dirname(eval_result_dir)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
        print('Directory created: ' + tmpdir)
    else:
        print('Directory already exists: ' + tmpdir)

    ndims = len(volshape)
    vol_num = len(test_paths)
    orig_lab_shape=nib.load(label_paths[0]).dataobj.shape
    num_lab = 1
    if len(orig_lab_shape)>3 and use_lab:
        num_lab = orig_lab_shape[-1]

    labshape = np.add(orig_lab_shape[:ndims] * np.array(rescale_factor_label), -2 * crop_sz*rescale_factor_label/rescale_factor).astype(int)  # [1:-1]

    #
    def img_sym_crop(img, crop_sz=crop_sz, img_sz=volshape, ndims=ndims):
        return img[:, crop_sz[0]:crop_sz[0] + img_sz[0], crop_sz[1]:crop_sz[1] + img_sz[1], ...] if ndims == 2 else img[:,crop_sz[0]:crop_sz[0] + img_sz[0], crop_sz[1]:crop_sz[1] + img_sz[1], crop_sz[2]:crop_sz[2] + img_sz[2], ...]

    def img_crop(img, crop_sz=crop_sz, img_sz=volshape, ndims=ndims, random_crop=False):
        if random_crop:
            return img[:, crop_sz[0][0]: img_sz[0] - crop_sz[1][0], crop_sz[0][1]: img_sz[1] - crop_sz[1][1],
                   ...] if ndims == 2 else img[:, crop_sz[0][0]: img_sz[0] - crop_sz[1][0],
                                           crop_sz[0][1]:img_sz[1] - crop_sz[1][1],
                                           crop_sz[0][2]: img_sz[2] - crop_sz[1][2], ...]
        else:
            return img_sym_crop(img, crop_sz=crop_sz, img_sz=img_sz, ndims=ndims)
            # return img_sym_crop(img, crop_sz=crop_sz[0], img_sz=img_sz, ndims=ndims)

    def preprocess(x, rand_int_scale=0., rescale_factor=rescale_factor,dimexpand=True,int_range=int_range):
        if rescale_factor != 1 and rescale_factor is not None:
            x = zoom(np.array(x), rescale_factor, mode='nearest')
        if rand_int_scale is None:
            return x
        x = np.minimum(x, int_range[1])
        x = np.maximum(x, int_range[0])
        if dimexpand:
            x=np.expand_dims(x,-1)
        if rand_int_scale > 0:
            return (x - np.min(x)) / (np.ptp(x) + eps) * (1. + np.random.uniform(-rand_int_scale, rand_int_scale))
        else:
            return (x - np.min(x)) / (np.ptp(x) + eps)


    def data_generator(idx1,idx2,path, lab_path=label_paths, batch_size=batch_size, pair_type=pair_type, use_lab=use_lab, crop_sz=crop_sz,
                           random_crop=False, rescale_factor=rescale_factor,rescale_factor_label=rescale_factor_label,img_instead_lab=False,num_lab=num_lab):

        [src_path, mv_path] = [path[0], path[1]] if pair_type == 'paired' else [path, path]
        if lab_path is not None:
            [src_lab_path, mv_lab_path] = [lab_path[0], lab_path[1]] if pair_type == 'paired' else [lab_path, lab_path]
        vol_shape_wo_crop = np.round(nib.load(path[0]).dataobj.shape * np.array(rescale_factor)).astype(int)  # [1:-1]
        vol_shape = (vol_shape_wo_crop - 2 * crop_sz)
        ndims = len(vol_shape)
        zero_phi = np.zeros([batch_size, *vol_shape, ndims])
        idx=list(range(batch_size))
        idx_mat=[[i,j] if i!=j else None for i in idx for j in idx]
        idx_mat.remove(None)

        # if img_instead_lab:
        #     use_lab=True
        #     num_lab=1
        if not use_lab:
            [src_lab_path, mv_lab_path]=[src_path,mv_path]

        if 1:
            crop = crop_sz
            crop_lab = [int(cs * rescale_factor_label // rescale_factor) for cs in crop]

            tgt_images = img_crop(
                np.stack([preprocess(nib.load(mv_path[id]).dataobj) for id in idx1], 0),
                crop_sz=crop, img_sz=volshape, random_crop=random_crop)

            tgt_labels = img_crop(np.stack(
                    [preprocess(nib.load(mv_lab_path[id]).dataobj, rand_int_scale=0,rescale_factor=rescale_factor_label,dimexpand=False) for id in idx1], 0), crop_sz=crop_lab, random_crop=random_crop, img_sz=labshape)

            idx2 = idx1 if pair_type == 'paired' else idx2
            src_images = img_crop(
                np.stack([preprocess(nib.load(src_path[id]).dataobj) for id in idx2], 0),
                crop_sz=crop, random_crop=random_crop, img_sz=volshape)

            src_labels = img_crop(np.stack(
                    [preprocess(nib.load(src_lab_path[id]).dataobj, rand_int_scale=0,rescale_factor=rescale_factor_label,dimexpand=False) for id in idx2], 0), crop_sz=crop_lab, random_crop=random_crop, img_sz=labshape)
            if num_lab<=1:
                tgt_labels =np.expand_dims(tgt_labels,-1)
                src_labels =np.expand_dims(src_labels,-1)
            inputs = [src_images, tgt_images, src_labels]
            outputs = [tgt_images, zero_phi, tgt_labels]
        # else:
        #     crop = crop_sz
        #     tgt_images = img_crop(
        #         np.stack([preprocess(nib.load(mv_path[id]).dataobj) for id in idx1], 0),
        #         crop_sz=crop, random_crop=random_crop, img_sz=volshape)
        #     idx2 = idx1 if pair_type == 'paired' else idx2
        #     src_images = img_crop(
        #         np.stack([preprocess(nib.load(src_path[id]).dataobj) for id in idx2], 0),
        #         crop_sz=crop, random_crop=random_crop, img_sz=volshape)
        #     inputs = [src_images, tgt_images]
        #     outputs = [tgt_images, zero_phi]
        return inputs,outputs


    # network
    def build_backbone(volshape=volshape,num_lab=num_lab,rescale_factor=rescale_factor,use_lab=use_lab):
        stn = func.networks.STN(volshape, name='image_warping', padding=1, use_aff=False, interp_method="linear")
        stn_syn = func.networks.STN(volshape, name='syn_image_warping', padding=1, use_aff=False,
                                    interp_method="linear")
        stn_lab = func.networks.STN(labshape, volshape, vol_feats=num_lab, name='label_warping', padding=1,
                                    use_aff=False, upsample_sz=int(rescale_factor_label // rescale_factor),
                                    interp_method='nearest' if use_lab else 'linear')
        disp_conv = tf.keras.layers.Conv3D(ndims, kernel_size=ndims, padding='same', name='disp')
        # net = net_core(volshape, nb_enc_features, nb_dec_features)
        net = net_core(volshape)
        return [net, disp_conv, stn, stn_syn, stn_lab]

    ##############################################################################################
    if 1:
        tf.reset_default_graph()
        [net, disp_conv, stn, _, stn_lab] = build_backbone(num_lab=num_lab)
        # -------------graph 1-------------
        # build graph
        input_src_image = tf.placeholder(tf.float32, [None, *volshape, 1])  # [-1] + volshape + [1])
        input_tgt_image = tf.placeholder(tf.float32, [None, *volshape, 1])

        net_out_syn = net([input_src_image, input_tgt_image])
        pred_disp = disp_conv(net_out_syn)  #*0
        pred_img = stn([input_src_image, pred_disp])

        # if use_lab:

        input_src_label = tf.placeholder(tf.float32, [None, *labshape, num_lab])
        pred_lab = stn_lab([input_src_label, pred_disp])

        infer_generator = data_generator

        saver = tf.train.Saver(max_to_keep=1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # save_path = model_path + "_1.tf"
        save_path = model_path + "_3.tf"
        saver.restore(sess, save_path)

        for idx1 in range(vol_num):  #
            idx_list=list(range(idx1)) + list(range(idx1 + 1, vol_num))
            for idx2 in idx_list:  #
                infer_input, infer_output = infer_generator([idx1], [idx2], test_paths,crop_sz=crop_sz, batch_size=batch_size,img_instead_lab=True)

                start = time.time()
                # if use_lab:
                inferFeed = {input_src_image: infer_input[0],
                           input_tgt_image: infer_input[1],
                           input_src_label: infer_input[2],
                           }
                disp_pred, img_pred, img_targ, lab_pred = sess.run(
                    [pred_disp, pred_img, input_tgt_image, pred_lab],
                    feed_dict=inferFeed)
                if use_lab:
                    # write_image(lab_pred[0, ...], file_path=warped_img_dir,
                    #             file_prefix='lab_warped_' + model_name + '_' + str(idx1) + '_from_' + str(idx2))
                    # write_image(infer_output[-1][0, ..., 0], file_path=warped_img_dir,
                    #             file_prefix='lab_target_' + str(idx1)) if idx2 == 0 or idx2 == idx1 + 1 else None
                    write_image(np.sum(lab_pred[0, ...],axis=-1), file_path=warped_img_dir,
                                file_prefix='lab_warped_' + model_name + '_' + str(idx1) + '_from_' + str(idx2))
                    write_image(np.sum(infer_output[-1][0, ..., 0],axis=-1), file_path=warped_img_dir,
                                file_prefix='lab_target_' + str(idx1)) if idx2 == 0 or idx2 == idx1 + 1 else None
                else:
                    write_image(lab_pred[0, ..., 0], file_path=warped_img_dir,
                                file_prefix='img_warped_' + model_name + '_' + str(idx1) + '_from_' + str(idx2))
                    write_image(infer_output[-1][0, ..., 0], file_path=warped_img_dir,
                                file_prefix='img_target_' + str(idx1)) if idx2 == 0 or idx2 == idx1 + 1 else None
                write_image(disp_pred[0, ...], file_path=warped_img_dir,
                            file_prefix='disp3_target_' + model_name + '_' + str(idx1) + '_from_' + str(idx2))
                if 0:
                    inferFeed = {input_src_image: infer_input[0],
                               input_tgt_image: infer_input[1],
                               # input_syn_disp: infer_input[1],
                               }
                    disp_pred, img_pred, img_targ = sess.run(
                        [pred_disp, pred_img, input_tgt_image],
                        feed_dict=inferFeed)
                    write_image(img_pred[0, ..., 0], file_path=warped_img_dir,
                                file_prefix='img_warped_' + model_name + '_' + str(idx1) + '_from_' + str(idx2))
                    write_image(infer_output[0][0, ..., 0], file_path=warped_img_dir,
                                file_prefix='img_target_' + str(idx1)) if idx2 == 0 or idx2 == idx1 + 1 else None
                    write_image(disp_pred[0, ...], file_path=warped_img_dir,
                                file_prefix='disp3_target_' + model_name + '_' + str(idx1) + '_from_' + str(idx2))
                end = time.time()
                time_cost = max(0., end - start)
                print('time cost:', time_cost)

        sess.close()
