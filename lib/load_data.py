from builtins import print

import numpy as np
import torch

from .load_dnerf import load_dnerf_data
from .load_hyper import Load_hyper_data


def load_data(args):

    K, depths = None, None
    times=None

    if args.dataset_type == 'hyper_dataset':
        data_class=Load_hyper_data(datadir=args.datadir,
                                    use_bg_points=args.use_bg_points, add_cam=args.add_cam)
        data_dict = dict(
            data_class=data_class,
            near=data_class.near, far=data_class.far,
            i_train=data_class.i_train, i_val=data_class.i_test, i_test=data_class.i_test,)
        return data_dict

    elif args.dataset_type == 'dnerf':
        images, poses, times, render_poses, render_times, hwf, i_split = load_dnerf_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near = 2.
        far = 6.
        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]
    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K
    render_poses = render_poses[...,:4]
    if times is None:
        times = torch.zeros(images.shape[0])
        render_times = torch.zeros(render_poses.shape[0])
    print('all_idx', images.shape[0])
    print('i_train=', i_train)
    print('i_test=', i_test)
    data_dict = dict(
            hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
            i_train=i_train, i_val=i_val, i_test=i_test,
            poses=poses, render_poses=render_poses,
            images=images, depths=depths,
            irregular_shape=irregular_shape, times=times, render_times=render_times,)

    return data_dict


