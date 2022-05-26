import warnings

warnings.filterwarnings("ignore")

import json
import os
import random

import numpy as np
import torch
from PIL import Image


class Load_hyper_data():
    def __init__(self, 
                 datadir, 
                 ratio=0.5,
                 use_bg_points=False,
                 add_cam=False):
        from .utils import Camera
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)

        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']

        self.add_cam = False
        if len(self.val_id) == 0:
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if
                            (i%4 == 0)])
            self.i_test = self.i_train+2
            self.i_test = self.i_test[:-1,]
        else:
            self.add_cam = True
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(i)
                if id in self.train_id:
                    self.i_train.append(i)
        assert self.add_cam == add_cam
        
        print('self.i_train',self.i_train)
        print('self.i_test',self.i_test)
        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['time_id'] for i in self.all_img]
        max_time = max(self.all_time)
        self.all_time = [meta_json[i]['time_id']/max_time for i in self.all_img]
        self.selected_time = set(self.all_time)
        self.ratio = ratio


        # all poses
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            camera = camera.scale(ratio)
            camera.position = camera.position - self.scene_center
            camera.position = camera.position * self.coord_scale
            self.all_cam_params.append(camera)

        self.all_img = [f'{datadir}/rgb/{int(1/ratio)}x/{i}.png' for i in self.all_img]
        self.h, self.w = self.all_cam_params[0].image_shape

        self.use_bg_points = use_bg_points
        if use_bg_points:
            with open(f'{datadir}/points.npy', 'rb') as f:
                points = np.load(f)
            self.bg_points = (points - self.scene_center) * self.coord_scale
            self.bg_points = torch.tensor(self.bg_points).float()
        print(f'total {len(self.all_img)} images ',
                'use cam =',self.add_cam, 
                'use bg_point=',self.use_bg_points)

    def load_idx(self, idx,not_dic=False):

        all_data = self.load_raw(idx)
        if not_dic == True:
            rays_o = all_data['rays_ori']
            rays_d = all_data['rays_dir']
            viewdirs = all_data['viewdirs']
            rays_color = all_data['rays_color']
            return rays_o, rays_d, viewdirs,rays_color
        return all_data

    def load_raw(self, idx):
        image = Image.open(self.all_img[idx])
        camera = self.all_cam_params[idx]
        pixels = camera.get_pixel_centers()
        rays_dir = torch.tensor(camera.pixels_to_rays(pixels)).float().view([-1,3])
        rays_ori = torch.tensor(camera.position[None, :]).float().expand_as(rays_dir)
        rays_color = torch.tensor(np.array(image)).view([-1,3])/255.
        return {'rays_ori': rays_ori, 
                'rays_dir': rays_dir, 
                'viewdirs':rays_dir / rays_dir.norm(dim=-1, keepdim=True),
                'rays_color': rays_color, 
                'near': torch.tensor(self.near).float().view([-1]), 
                'far': torch.tensor(self.far).float().view([-1]),}
