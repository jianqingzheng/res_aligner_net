import numpy as np
import nibabel as nib
import os

import pyquaternion as quater
import scipy.ndimage as spimg
from scipy.ndimage import zoom
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff,cdist

ndims=3
# conn=[2]
# mask_is_binary=False
conn=[1,2,2,1]
mask_is_binary=True
def generate_struct_kernel(conn=conn,ndims=ndims):
    kern = np.zeros([2*len(conn)+1]*ndims)
    loc=[len(conn)]*ndims
    kern[loc]=1
    for c in conn:
        se=ndimage.generate_binary_structure(ndims, c)
        kern=ndimage.binary_dilation(kern, structure=se).astype(kern.dtype)
    return kern

struct_kernel=generate_struct_kernel()


def rigid_transform_3D(A, B):
    # assert A.shape == B.shape
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)
    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return R, t

def dilate_vol(vol,structure=struct_kernel,iterations=1):
    return ndimage.binary_dilation(vol,structure=structure,iterations=iterations)

def ddf_decouple(ddf_vol,masks,ndims=3,sample_rate=7,dilate_iter=4,mask_is_binary=mask_is_binary):
    '''
    function [params, deform]=ddf_vol2params(ddf_vol,masks)
    num_dims=ndims(ddf_vol);
    index = cell(1, num_dims);
    index(:) = {':'};
    params=[];
    deform=0;
    for i =1:size(masks,num_dims)
        index{end} = i;
        [param,ddf_def]=ddf_vol2param(ddf_vol,masks(index{:}));
        params=cat(3,params,param);
        deform=deform+ddf_def;
    end
    end
    '''
    if mask_is_binary:
        masks=np.sum(masks,axis=-1,keepdims=True)

    size_vol=masks.shape
    ddf_rigid = np.zeros_like(ddf_vol)
    masks_ctl = masks
    boundary_ignore=16
    masks_ctl[:,boundary_ignore:size_vol[1]-boundary_ignore,boundary_ignore:size_vol[2]-boundary_ignore,boundary_ignore:size_vol[3]-boundary_ignore,:]=0
    masks = np.ones_like(masks)
    for b in range(size_vol[0]):

        mask1=masks[b,...]
        mask2=masks_ctl[b,...]
        mask1_comp=np.logical_not(np.any(mask1,axis=-1,keepdims=True))
        plus=0
        if np.any(mask1_comp):
            mask1 = np.concatenate([mask1,mask1_comp],axis=-1)
            mask2 = np.concatenate([mask2,mask1_comp],axis=-1)
            plus=1
        for m in range(size_vol[-1]+plus):
            _,_,ddf_r=ddf2param(ddf_vol[b,...],mask1[...,m],mask2[...,m],ndims=ndims,sample_rate=sample_rate)
            ddf_rigid[b,...]+=ddf_r
    ddf_deform=ddf_vol-ddf_rigid
    return ddf_rigid,ddf_deform

def ddf2param(ddf_vol,msk_slv=None,msk_ctl=None,ndims=3,sample_rate=13):
    '''
    function [param,deform]=ddf_vol2param(ddf_vol,mask)
    num_dims=ndims(ddf_vol)-1;
    [ind,coord]=lab2ind(mask);
    ddf_reshape=reshape(ddf_vol,[numel(ddf_vol)/num_dims,num_dims]);
    ddf=ddf_reshape(ind,:);
    coord_tgt=coord+ddf;
    [R,t] = rigid_transform_3D(coord', coord_tgt');
    param=[R,t];
    coord_rig=(R*coord'+t)';
    ddf_def=coord_tgt-coord_rig;
    deform_res=zeros(size(ddf_reshape));
    deform_res(ind,:)=ddf_def;
    deform=reshape(deform_res,size(ddf_vol));
    end
    '''
    orig_shape=ddf_vol.shape
    index_slv=np.nonzero(msk_slv)
    index_ctl = np.nonzero(msk_ctl)

    coord_slv_src = np.stack(index_slv,axis=0)
    coord_ctl_src = np.stack(index_ctl, axis=0)

    coord_slv_tgt = coord_slv_src + np.stack([ddf_vol[(*index_slv,np.ones_like(index_slv[0])*d)] for d in range(orig_shape[-1])],axis=0)
    coord_ctl_tgt = coord_ctl_src + np.stack([ddf_vol[(*index_ctl, np.ones_like(index_ctl[0]) * d)] for d in range(orig_shape[-1])], axis=0)

    R, t = rigid_transform_3D(coord_ctl_src[...,::sample_rate], coord_ctl_tgt[...,::sample_rate])
    # coord_rig= np.transpose((R@np.transpose(coord_src)) + t)
    coord_rig = (R @ coord_slv_src) + t
    ddf_def = np.zeros_like(ddf_vol)
    ddf_rig = np.zeros_like(ddf_vol)
    for d in range(orig_shape[-1]):
        ddf_def[(*index_slv, np.ones_like(index_slv[0]) * d)] = coord_slv_tgt[d, ...] - coord_rig[d, ...]
        ddf_rig[(*index_slv, np.ones_like(index_slv[0]) * d)] = coord_rig[d, ...] - coord_slv_src[d, ...]

    param=np.concatenate([R,t],axis=1)
    return param,ddf_def,ddf_rig
#

def write_image(data, file_path=None, file_prefix=''):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print('Directory created: ' + file_path)
    # else:
    #     print('Directory already exists: ' + file_path)

    if file_path is not None:
        # batch_size = input_.shape[0]
        affine = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        nib.save(nib.Nifti1Image(data, affine),
                 os.path.join(file_path, file_prefix))



# grid option

def random_ddf(batch_sz, img_sz, pn_spline=20, pn_gauss=10, range_spline=2., range_gauss=24, spread_range=[5., 16.],
               transl_range=5., rot_range=np.pi / 4):
    rot_df = get_rot_ddf(img_sz, vec=np.random.uniform(-1., 1., [batch_sz, 3]),
                         ang=np.random.uniform(-rot_range, rot_range, [batch_sz]))
    ddf0 = np.tile([generate_random_gaussian_ddf(img_sz, pn_gauss, range_sz=range_gauss, spread_std=spread_range) \
                    + np.random.uniform(-transl_range, transl_range, [3])], [batch_sz, 1, 1, 1, 1]) \
           + rot_df

    def boundary_replicate(sample_coords, input_size, padding=5):
        return np.stack(
            [np.maximum(np.minimum(sample_coords[..., i], input_size[i] - 1 + padding), 0 - padding) for i in
             range(len(input_size))], axis=-1), \
               np.prod([((sample_coords[..., i] < input_size[i]) * (sample_coords[..., i] >= 0)) for i in
                        range(len(input_size))], axis=0)

    ref = get_reference_grid(img_sz)
    cf1, ind = boundary_replicate(ddf0 + ref, img_sz)
    return cf1 - ref, np.expand_dims(ind, -1), rot_df


def generate_random_gaussian_ddf(img_sz, pn=30, range_sz=5, spread_std=[0.1, 1.]):

    x = np.floor(np.random.uniform(range_sz / 2., img_sz[0] - range_sz / 2., [1, pn])).astype('int')
    y = np.floor(np.random.uniform(range_sz / 2., img_sz[1] - range_sz / 2., [1, pn])).astype('int')
    z = np.floor(np.random.uniform(range_sz / 2., img_sz[2] - range_sz / 2., [1, pn])).astype('int')

    odf = np.random.uniform(-range_sz, range_sz, [pn, 3])
    vol = np.zeros([img_sz[0], img_sz[1], img_sz[2], 3])
    vol[x, y, z] = odf
    return spimg.gaussian_filter(vol, np.random.uniform(spread_std[0], spread_std[1]))


def get_rot_ddf(grid_size, vec=[[0., 0., 1.]], ang=[[0.]]):
    vec = np.array(vec)
    ang = np.array(ang)
    batch_num = ang.shape[0]
    ref_grids = get_reference_grid(grid_size,
                                   bias_scale=1.)  # [get_reference_grid(grid_size, bias_scale=1.) for i in range(batch_num)]
    return np.reshape(np.matmul(np.reshape(np.tile(ref_grids, [batch_num, 1, 1, 1, 1]), [batch_num, -1, 3]),
                                vecang2rotmats(vec, ang)), [batch_num] + grid_size + [3]) - ref_grids


def get_reference_grid(grid_size, bias_scale=0.):
    return np.stack(np.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=-1).astype('float') - bias_scale * (np.array(grid_size) - 1) / 2.


def resample_linear(inputs, ddf=None, sample_coords=None,random_boundary=True):
    if random_boundary:
        random_factor = np.random.uniform(0., 1.)
        min_val = np.min(inputs)
        inputs[:, 0, :, :] = min_val * random_factor + (1 - random_factor) * inputs[:, 0, :, :]
        inputs[:, -1, :, :] = min_val * random_factor + (1 - random_factor) * inputs[:, -1, :, :]
        inputs[:, :, 0, :] = min_val * random_factor + (1 - random_factor) * inputs[:, :, 0, :]
        inputs[:, :, -1, :] = min_val * random_factor + (1 - random_factor) * inputs[:, :, -1, :]
        inputs[:, :, :, 0] = min_val * random_factor + (1 - random_factor) * inputs[:, :, :, 0]
        inputs[:, :, :, -1] = min_val * random_factor + (1 - random_factor) * inputs[:, :, :, -1]

    input_size = inputs.shape[1:4]
    sample_coords = get_reference_grid(input_size) + ddf if sample_coords is None else sample_coords
    spatial_rank = 3  # inputs.ndim - 2
    xy = [sample_coords[..., i] for i in
          range(sample_coords.shape[-1])]  # tf.unstack(sample_coords, axis=len(sample_coords.shape)-1)
    index_voxel_coords = [np.floor(x) for x in xy]

    def boundary_replicate(sample_coords0, input_size0, plus=0):
        return np.maximum(np.minimum(sample_coords0, input_size0 - 2 + plus), 0 + plus)

    def boundary_replicate_float(sample_coords0, input_size0, plus=0.):
        return np.maximum(np.minimum(sample_coords0, input_size0 - 1 + plus), 0 + plus)

    xy = [boundary_replicate_float(x.astype('float32'), input_size[idx]) for idx, x in enumerate(xy)]
    spatial_coords = [boundary_replicate(x.astype('int32'), input_size[idx])
                      for idx, x in enumerate(index_voxel_coords)]
    spatial_coords_plus1 = [boundary_replicate((x + 1).astype('int32'), input_size[idx], 1)
                            for idx, x in enumerate(index_voxel_coords)]

    weight = [np.expand_dims(x - i.astype('float32'), -1) for x, i in zip(xy, spatial_coords)]
    weight_c = [np.expand_dims(i.astype('float32') - x, -1) for x, i in zip(xy, spatial_coords_plus1)]

    sz = list(spatial_coords[0].shape)
    batch_coords = np.tile(np.reshape(range(sz[0]), [sz[0]] + [1] * (len(sz) - 1)), [1] + sz[1:])
    sc = (spatial_coords, spatial_coords_plus1)
    binary_codes = [[int(c) for c in format(i, '0%ib' % spatial_rank)] for i in range(2 ** spatial_rank)]

    make_sample = lambda bc: inputs[batch_coords, sc[bc[0]][0], sc[bc[1]][1], sc[bc[2]][
        2], ...]  # tf.gather_nd(inputs, np.stack([batch_coords] + [sc[c][i] for i, c in enumerate(bc)], -1))
    samples = [make_sample(bc) for bc in binary_codes]

    def pyramid_combination(samples0, weight0, weight_c0):
        if len(weight0) == 1:
            return samples0[0] * weight_c0[0] + samples0[1] * weight0[0]
        else:
            return pyramid_combination(samples0[::2], weight0[:-1], weight_c0[:-1]) * weight_c0[-1] + \
                   pyramid_combination(samples0[1::2], weight0[:-1], weight_c0[:-1]) * weight0[-1]

    return pyramid_combination(samples, weight, weight_c)


def vecang2rotmats(vec, ang):
    return np.stack([np.reshape(vecang2rotmat(vec[i, ...], ang[i, ...]), [3, 3]) for i in range(len(vec))], 0)


def vecang2rotmat(vec, ang):
    q = quater.Quaternion(axis=vec, angle=ang)
    return q.rotation_matrix

# dataloader



def real_data_generator(path, batch_size=3, pair_type='unpaired', random_crop=True, rescale_factor=0.5, crop_sz=[0,0,0],int_range=[-100,300], num_lab=9):
    [src_path, tgt_path] = [path[0], path[1]] if pair_type == 'paired' else [path, path]
    vol_shape_wo_crop = np.round(nib.load(path[0]).dataobj.shape * np.array(rescale_factor)).astype(int)  # [1:-1]
    vol_shape = (vol_shape_wo_crop - 2 * crop_sz)
    vol_num = len(src_path)
    ndims = len(vol_shape)
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    int_scale=.2

    while True:
        rand_int_scale = np.random.uniform(-int_scale, int_scale)
        crop = [np.random.randint(0., 2 * cs) for cs in crop_sz] if random_crop else crop_sz
        crop = [crop, [2 * cs - c for cs, c in zip(crop_sz, crop)]]
        idx1 = np.random.randint(0, vol_num, size=batch_size)
        tgt_images = img_crop(
            (np.stack([data_preprocess(nib.load(tgt_path[id]).dataobj, rand_int_scale=rand_int_scale,int_range=int_range) for id in idx1],
                      0)),
            crop_sz=crop, random_crop=random_crop, img_sz=vol_shape_wo_crop)
        # tgt_images = img_crop(np.expand_dims(np.stack([unit_normalize(nib.load(tgt_path[id]).dataobj,rand_int_scale=rand_int_scale) for id in idx1], 0), -1),crop_sz=crop, random_crop=random_crop, img_sz=vol_shape_wo_crop)
        idx2 = idx1 if pair_type == 'paired' else np.random.randint(0, vol_num, size=batch_size)
        src_images = img_crop(
            (np.stack([data_preprocess(nib.load(src_path[id]).dataobj, rand_int_scale=rand_int_scale,int_range=int_range) for id in idx2],
                      0)),
            crop_sz=crop, random_crop=random_crop, img_sz=vol_shape_wo_crop)
        inputs = [src_images, tgt_images]
        outputs = [tgt_images, zero_phi]
        yield inputs, outputs

def syn_data_generator(path, batch_size=3, rec_loss=False, num_scale=1, random_crop=True, crop_sz=[0,0,0],
                       rescale_factor=1, syn_rot_range=np.pi/6,int_range=[-100,300]):
    vol_shape_wo_crop = np.round(nib.load(path[0]).dataobj.shape * np.array(rescale_factor)).astype(int)  # [1:-1]
    vol_shape = (vol_shape_wo_crop - 2 * crop_sz)
    vol_num = len(path)

    while True:
        crop = [np.random.randint(0. * cs, 2. * cs) for cs in crop_sz] if random_crop else crop_sz
        crop = [crop, [2 * cs - c for cs, c in zip(crop_sz, crop)]]
        idx1 = np.random.randint(0, vol_num, size=batch_size)
        orig_images = np.stack([data_preprocess(nib.load(path[id]).dataobj,int_range=int_range) for id in idx1], 0)
        aug_phi, _, _ = random_ddf(batch_size, img_sz=list(vol_shape_wo_crop), transl_range=1., rot_range=np.pi)

        src_images = img_crop(resample_linear(orig_images, aug_phi), crop_sz=crop, img_sz=vol_shape_wo_crop,
                                random_crop=random_crop)
        syn_phi, _, _ = random_ddf(batch_size, img_sz=list(vol_shape), transl_range=8., rot_range=syn_rot_range)

        inputs = [src_images, syn_phi]
        outputs = [np.zeros([batch_size, *vol_shape, 1]), syn_phi] if rec_loss else [syn_phi] * num_scale
        yield inputs, outputs

def img_sym_crop(img, crop_sz=None, img_sz=None, ndims=ndims):
    return img[:, crop_sz[0]:crop_sz[0] + img_sz[0], crop_sz[1]:crop_sz[1] + img_sz[1], ...] if ndims == 2 else img[
                                                                                                                :,
                                                                                                                crop_sz[
                                                                                                                    0]:
                                                                                                                crop_sz[
                                                                                                                    0] +
                                                                                                                img_sz[
                                                                                                                    0],
                                                                                                                crop_sz[
                                                                                                                    1]:
                                                                                                                crop_sz[
                                                                                                                    1] +
                                                                                                                img_sz[
                                                                                                                    1],
                                                                                                                crop_sz[
                                                                                                                    2]:
                                                                                                                crop_sz[
                                                                                                                    2] +
                                                                                                                img_sz[
                                                                                                                    2],
                                                                                                                ...]

def img_crop(img, crop_sz=[0,0,0], img_sz=None, ndims=3, random_crop=True):
    if random_crop:
        return img[:, crop_sz[0][0]: img_sz[0] - crop_sz[1][0], crop_sz[0][1]: img_sz[1] - crop_sz[1][1],
               ...] if ndims == 2 else img[:, crop_sz[0][0]: img_sz[0] - crop_sz[1][0],
                                       crop_sz[0][1]:img_sz[1] - crop_sz[1][1],
                                       crop_sz[0][2]: img_sz[2] - crop_sz[1][2], ...]
    else:
        return img_sym_crop(img, crop_sz=crop_sz, img_sz=img_sz, ndims=ndims)

def data_preprocess(x,rand_int_scale=0., rescale_factor=1,num_lab=None, vol_shape=[64,64,64], dimexpand=True,int_range=[-100,300],eps=1e-5):
    if rescale_factor != 1 and rescale_factor is not None:
        x = zoom(np.array(x), rescale_factor, mode='nearest')

    if num_lab is None:
        x = np.minimum(x, int_range[1])
        x = np.maximum(x, int_range[0])
        if dimexpand:
            x=np.expand_dims(x,-1)
        if rand_int_scale is not None:
            return (x - np.min(x)) / (np.ptp(x) + eps) * (1. + rand_int_scale)
        else:
            return (x - np.min(x)) / (np.ptp(x) + eps)
    else:
        if num_lab<=1:
            if dimexpand:
                x=np.expand_dims(x,-1)
        return x

# evaluation

def eval_dsc(vol1, vol2,disp=None, thresh=0.1):
    eps = 10 ** -5
    bv1 = vol1 > thresh
    bv2 = vol2 > thresh
    intersection = np.logical_and(bv1, bv2)
    return 2. * intersection.sum() / (bv1.sum() + bv2.sum() + eps)

def eval_hd(vol1,vol2,disp=None,thresh=0.1):
    return max(directed_hausdorff(np.argwhere(vol1>thresh), np.argwhere(vol2>thresh))[0],directed_hausdorff(np.argwhere(vol2>thresh), np.argwhere(vol1>thresh))[0])

def eval_asd(vol1,vol2,disp=None,thresh=0.1):
    rescale_factor=.5
    # vol1 = zoom(vol1, [rescale_factor] * ndims + [1], mode='nearest')
    # vol2 = zoom(vol2, [rescale_factor] * ndims + [1], mode='nearest')
    vol1 = zoom(vol1, [rescale_factor] * ndims, mode='nearest')
    vol2 = zoom(vol2, [rescale_factor] * ndims, mode='nearest')
    dist=cdist(np.argwhere(ndimage.laplace(vol1) > thresh), np.argwhere(ndimage.laplace(vol2) > thresh),'euclidean')
    if 0 == dist.shape[1]:
        return 0
    elif 0 == dist.shape[0]:
        return min(vol1.shape)
    else:
        return np.mean(np.concatenate([np.min(dist,axis=-1),np.min(dist,axis=-2)],axis=0))/rescale_factor

def eval_detJ_lab(vol1=None,vol2=None,disp=None,thresh=0.5):
    ndims=3
    label=vol1>thresh
    label=label*(ndimage.laplace(label) < 0.1)
    rescale_factor=2
    label=label[...,::rescale_factor,::rescale_factor,::rescale_factor]

    Jacob=np.stack(np.gradient(disp,axis=[-4,-3,-2]),-1)
    Jacob[..., 0, 0] = Jacob[..., 0, 0] + 1
    Jacob[..., 1, 1] = Jacob[..., 1, 1] + 1
    Jacob[..., 2, 2] = Jacob[..., 2, 2] + 1
    return np.sum((np.linalg.det(Jacob)<0)*label)
