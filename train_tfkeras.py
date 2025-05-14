import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import glob
from utils import *
import datetime

import ran_func as func


eps = 1e-5


def train(net_core=None, model_path=None, crop_sz=None, rescale_factor=None, num_lab=None, restore_pre_train=None,
          pair_type=None, train_path=None, batch_size=None, nb_epochs=None, train_stages=[], grad_weight=None, thresh=None,int_range=None):

    # restore_pre_train = False
    summary_error = False
    random_crop = True
    steps_per_epoch = 1000
    suffix='*.nii.gz'
    # suffix='*.nii'
    process_suffix = '_x2'
    # num_scale = 5

    if pair_type == 'paired':
        fx_pth_train = glob.glob(os.path.join(train_path, 'fixed_images'+process_suffix, suffix))
        mv_pth_train = glob.glob(os.path.join(train_path, 'moving_images'+process_suffix, suffix))
        train_paths = [mv_pth_train, fx_pth_train]
        syn_paths = mv_pth_train + fx_pth_train
        volshape = np.add(nib.load(fx_pth_train[0]).dataobj.shape * np.array(rescale_factor), -2 * crop_sz).astype(int)
    else:
        train_paths = glob.glob(os.path.join(train_path, 'images'+process_suffix, suffix))
        syn_paths = train_paths
        # label_paths = glob.glob(os.path.join(train_path, 'labels', suffix))
        volshape = np.add(nib.load(train_paths[0]).dataobj.shape * np.array(rescale_factor), -2 * crop_sz).astype(int)
        # vol_shape_wo_rsz = np.round(nib.load(train_paths[0]).dataobj.shape).astype(int)  # [1:-1]
    ndims = len(volshape)

    # network
    def build_backbone(volshape=volshape):
        stn = func.networks.STN(volshape, name='image_warping', padding=1, use_aff=False,interp_method="linear")
        stn_syn = func.networks.STN(volshape, name='syn_image_warping', padding=1, use_aff=False,interp_method="linear")
        disp_conv = tf.keras.layers.Conv3D(ndims, kernel_size=ndims, padding='same', name='disp')
        # net = net_core(volshape, nb_enc_features, nb_dec_features)
        net = net_core(volshape)
        return [net, disp_conv, stn, stn_syn]

    ##############################################################################################

    tf.reset_default_graph()
    [net, disp_conv, stn, stn_syn] = build_backbone()
    if 1 in train_stages:
        # -------------graph-------------
        # build graph
        input_src_image = tf.placeholder(tf.float32, [None, *volshape, 1])  # [-1] + volshape + [1])
        input_syn_disp = tf.placeholder(tf.float32, [None, *volshape, 3])  # [-1] + volshape + [3])
        syn_tgt_img = stn_syn([input_src_image, input_syn_disp])
        net_out_syn = net([input_src_image, syn_tgt_img])
        pred_disp = disp_conv(net_out_syn)
        pred_img = stn([input_src_image, pred_disp])
        train_generator = syn_data_generator(syn_paths, batch_size=batch_size, crop_sz=crop_sz, rec_loss=False, num_scale=1, random_crop=random_crop, rescale_factor=rescale_factor, syn_rot_range=np.pi/1,int_range=int_range)

        # -------------loss and optimizer-------------
        init_lr = 0.00001
        loss_total = 0
        nb_print_steps = 100
        init_disp_mse = func.losses.mse(0., input_syn_disp)
        loss_mse_img = func.losses.mse(pred_img, syn_tgt_img)
        loss_mse_disp = func.losses.mse(pred_disp, input_syn_disp)
        regular_smooth = func.losses.Grad()
        loss_grad_disp = regular_smooth.loss(y_pred=pred_disp)
        loss_total += .1 * loss_mse_img
        loss_total += 1 * loss_mse_disp
        loss_total += 1 * loss_grad_disp

        # train_op = tf.train.MomentumOptimizer(learning_rate=config['Train']['learning_rate'], momentum=0.9).minimize(loss_total)
        train_op = tf.train.AdamOptimizer(init_lr).minimize(loss_total)
        # train_op = tf.train.AdamOptimizer(config['Train']['learning_rate']).minimize(loss_total)

        # -------------restore-------------
        saver = tf.train.Saver(max_to_keep=1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        if restore_pre_train:
            saver.restore(sess, model_path + "_1.tf")
        save_path = model_path + "_1.tf"
        # summary
        if summary_error:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # time.asctime(time.gmtime())
            train_log_dir = os.path.join(model_path + '_log', current_time, 'train')
            valid_log_dir = os.path.join(model_path + '_log', current_time, 'valid')
            train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            # valid_summary_writer = tf.summary.FileWriter(valid_log_dir, sess.graph)

        # -------------training-------------
        loss_min = 1e10
        init_mse = 0
        for epoch in range(nb_epochs[0]):
            avg_mse_disp, avg_mse_img, avg_rate = 0, 0, 0
            for step in range(steps_per_epoch):
                train_input, train_output = next(train_generator)
                trainFeed = {input_src_image: train_input[0],
                             input_syn_disp: train_input[1],
                             }
                sess.run(train_op, feed_dict=trainFeed)

                if step % nb_print_steps == 0:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    # [loss] = sess.run([loss_total], feed_dict=trainFeed)#.as_type("float32")
                    loss, mse_disp, mse_img, init, grad_disp = sess.run([loss_total, loss_mse_disp, loss_mse_img, init_disp_mse,loss_grad_disp],
                                                             feed_dict=trainFeed)
                    avg_mse_disp = (avg_mse_disp * (step // nb_print_steps) + mse_disp * 1) / (
                            step // nb_print_steps + 1)
                    avg_rate = (avg_rate * (step // nb_print_steps) + (loss / (init + 1)) * 1) / (
                            step // nb_print_steps + 1)
                    avg_mse_img = (avg_mse_img * (step // nb_print_steps) + mse_img * 1) / (step // nb_print_steps + 1)
                    print(
                        'Epoch %d/%d Step %d [%s]: Loss:%f=%f(disp)+%f(grad)+%f(image), (init:%f); avg mse: disp=%f, img=%f, rate=%f' %
                        (epoch,nb_epochs[0], step,
                         current_time,
                         loss,
                         mse_disp,
                         grad_disp,
                         mse_img,
                         init,
                         avg_mse_disp,
                         avg_mse_img,
                         avg_rate,
                         ))
                    if summary_error:
                        train_summary = tf.compat.v1.Summary()
                        train_summary.value.add(tag='loss', simple_value=loss)
                        train_summary_writer.add_summary(train_summary, step)
                        train_summary_writer.flush()
                        del train_summary
                    if loss / (init + 50) <= loss_min / (init_mse + 50) and not np.isnan(loss):
                        loss_min = loss
                        init_mse = init
                        save_path = saver.save(sess, save_path, write_meta_graph=False)
                        print("Model saved in: %s" % save_path)
                    else:
                        print("minimal mse: %f, init mse: %f" % (loss_min, init_mse))
        sess.close()

    ##############################################################################################

    tf.reset_default_graph()
    [net, disp_conv, stn, _] = build_backbone()
    if 2 in train_stages:
        # -------------1.graph-------------
        input_src_image = tf.placeholder(tf.float32, [None, *volshape, 1])  # [-1] + volshape + [1])
        input_tgt_image = tf.placeholder(tf.float32, [None, *volshape, 1])

        net_out_syn = net([input_src_image, input_tgt_image])
        pred_disp = disp_conv(net_out_syn)
        pred_img = stn([input_src_image, pred_disp])

        train_generator = real_data_generator(train_paths, batch_size=batch_size,int_range=int_range, crop_sz=crop_sz)

        # -------------loss and optimizer-------------
        init_lr = 0.0001
        loss_total = 0
        nb_print_steps = 100

        if num_lab==1:
            loss_mse_img = func.losses.mse(pred_img, input_tgt_image,input_tgt_image>thresh)
            regular_smooth = func.losses.Grad(apear_scale=9)
        else:
            loss_img_sim = func.losses.NCC(win=[8] * ndims)
            loss_mse_img = loss_img_sim.loss(pred_img, input_tgt_image,input_tgt_image>thresh)
            regular_smooth = func.losses.Grad(apear_scale=25)


        loss_grad_disp = regular_smooth.loss(y_pred=pred_disp,img=input_tgt_image)

        loss_total += 2 * loss_mse_img
        loss_total += grad_weight * loss_grad_disp

        # train_op = tf.train.MomentumOptimizer(init_lr, momentum=0.9).minimize(loss_total)
        train_op = tf.train.AdamOptimizer(init_lr).minimize(loss_total)
        # train_op = tf.train.AdamOptimizer(init_lr).minimize(loss_total)

        # -------------restore-------------
        saver = tf.train.Saver(max_to_keep=1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        if restore_pre_train:
            saver.restore(sess, model_path + "_2.tf")
        else:
            saver.restore(sess, model_path + "_1.tf")
        save_path = model_path + "_2.tf"
        # summary
        if summary_error:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # time.asctime(time.gmtime())
            train_log_dir = os.path.join(model_path + '_log', current_time, 'train')
            # valid_log_dir = os.path.join(model_path + '_log', current_time, 'valid')
            train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            # valid_summary_writer = tf.summary.FileWriter(valid_log_dir, sess.graph)

        # -------------training-------------
        loss_min,grad_disp_min,mse_img_min = 1e10,1e10,1e10
        for epoch in range(nb_epochs[1]):
            avg_grad_disp, avg_mse_img = 0, 0
            for step in range(steps_per_epoch):
                train_input, train_output = next(train_generator)
                trainFeed = {input_src_image: train_input[0],
                             input_tgt_image: train_input[1],
                             }
                sess.run(train_op, feed_dict=trainFeed)

                if step % nb_print_steps == 0:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    # [loss] = sess.run([loss_total], feed_dict=trainFeed)#.as_type("float32")
                    loss, grad_disp, mse_img = sess.run([loss_total, loss_grad_disp, loss_mse_img], feed_dict=trainFeed)
                    avg_grad_disp = (avg_grad_disp * (step // nb_print_steps) + grad_disp * 1) / (
                            step // nb_print_steps + 1)
                    avg_mse_img = (avg_mse_img * (step // nb_print_steps) + (mse_img * 1)) / (
                                step // nb_print_steps + 1)
                    print('Epoch %d/%d Step %d [%s]: Loss:%f = %f (disp reg)+%f (img); avg: grd_disp=%f, mse_img=%f' %
                          (epoch,nb_epochs[1], step,
                           current_time,
                           loss,
                           grad_disp,
                           mse_img,
                           avg_grad_disp,
                           avg_mse_img,
                           ))
                    if summary_error:
                        train_summary = tf.compat.v1.Summary()
                        train_summary.value.add(tag='loss', simple_value=loss)
                        train_summary_writer.add_summary(train_summary, step)
                        train_summary_writer.flush()
                        del train_summary
                    if avg_mse_img <= loss_min * 1.0 and step>500 and not np.isnan(loss):
                        loss_min = avg_mse_img
                        grad_disp_min, mse_img_min=avg_grad_disp, avg_mse_img
                        save_path = saver.save(sess, save_path, write_meta_graph=False)
                        print("Model saved in: %s" % save_path)
                    else:
                        print("minimal Loss:%f = %f (disp reg)+%f (img)." % (loss_min,grad_disp_min,mse_img_min))
        sess.close()
