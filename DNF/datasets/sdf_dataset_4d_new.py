from __future__ import division

import glob
import sys
from torch.utils.data import Dataset
import os
import numpy as np
import trimesh
import torch
import json
from tqdm import tqdm
from timeit import default_timer as timer

class SDFDataset(Dataset):
    def __init__(
            self,
            data_dir='./DT4D',
            train_split = [],
            batch_size=64,
            num_workers=12,
            sample_info={},
            cache_data=True,
            only_shape=True,
            shape_ft=False,
            pose_ft=False,
            target_dir='',
            **kwargs
    ):

        ###################################################################################################
        # SDF
        ###################################################################################################
        sdf_samples_info = sample_info['sdf']
        self.num_points_sdf = sdf_samples_info['num_points']

        if self.num_points_sdf > 0:
            print()
            print("num sdf samples", self.num_points_sdf)
            print()

        ###################################################################################################
        # Flow
        ###################################################################################################
        sample_flow_info = sample_info['flow']
        self.num_points_flow = np.array(sample_flow_info['num_points'])

        self.num_flow_samples_list = []

        self.num_samples_per_shape = {
            'sdf': self.num_points_sdf,
            'flow': self.num_points_flow,
        }

        self.data_dir = data_dir
        # self.sample_paths = sorted(glob.glob(self.data_dir+'/*'))
        self.sample_paths = []
        self.instance_filenames = []
        self.train_split = train_split
        self.indice_list = []

        for name in train_split:
            filename = name.split('-')[1]
            self.sample_paths.append(os.path.join(self.data_dir, filename))
            self.indice_list.append(int(name.split('-')[0]))
        print("length of sample paths",len(self.sample_paths))

        self.cache_data = cache_data
        self.cache = []
        self.cache_tpose = []
        self.cache_flow = []
        self.shape_ft = shape_ft
        self.pose_ft = pose_ft
        self.only_shape = only_shape

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Preload data
        if self.cache_data:
            print("Preloading cached data ...")

            # T-Poses
            print("loading T-pose")
            for index in tqdm(range(len(self.sample_paths))):
                data = self.sample_paths[index]
                data_dict = self._load_sample(index, data, is_tpose=True)
                self.cache_tpose.append(data_dict)

                if not self.only_shape:
                    data_dict_flow = self._load_sample(index, self.sample_paths[index], is_tpose=False)
                    self.cache_flow.append(data_dict_flow)

            print("Length of data", len(self.cache_tpose))
            if not self.only_shape:
                print("Length of flow data", len(self.cache_flow))
            print("Loaded cached data ...")

        self.num_identities = self.indice_list[-1] + 1
        # self.num_identities = len(self.indice_list)

    def __len__(self):
        return len(self.sample_paths)

    def get_num_identities(self):
        return self.num_identities

    def get_num_samples_per_shape(self):
        return self.num_samples_per_shape

    def _load_sample(self, idx, data, is_tpose):
        shape_path = data

        # BOUNDARY
        # points_sdf_dict = {}
        sdf_list = []
        sdf_path = os.path.join(shape_path, 'points_seq')
        if is_tpose and self.num_points_sdf > 0:
            sdf_samples_path = os.path.join(sdf_path, '00000000.npz')
            sdf_samples_npz = np.load(sdf_samples_path)
            sdf_points = sdf_samples_npz['points']
            sdf_list.append(np.array(sdf_points, dtype=np.float32))

        # FLOW

        flow_list = []
        # normal_list = []

        flow_path = os.path.join(shape_path, 'flow_seq')
        if not is_tpose and self.num_points_flow > 0:
            flow_samples_path = os.path.join(flow_path, 't-pose_flow_samples.npz')
            flow_samples_npz = np.load(flow_samples_path)
            points_flow = flow_samples_npz['points']
            # normals_flow = flow_samples_npz['normals']
            # print('t', points_flow.shape)

            flow_list.append(np.array(points_flow, dtype=np.float32))
            # normal_list.append(np.array(normals_flow, dtype=np.float32))

            for idx in range(1, 16):
                flow_samples_path = os.path.join(flow_path, str(idx) + '-pose_flow_samples.npz')
                flow_samples_npz = np.load(flow_samples_path)
                points_flow = flow_samples_npz['points']
                # normals_flow = flow_samples_npz['normals']
                # print(idx, points_flow.shape)

                flow_list.append(np.array(points_flow, dtype=np.float32))
                # normal_list.append(np.array(normals_flow, dtype=np.float32))
            # print("loading", len(flow_list),flow_list[5].shape)

        return {
            'points_sdf': sdf_list,
            'points_flow': flow_list,
            # 'points_normal': normal_list,
            'path': shape_path,
            'identity_id': idx,
        }

    def _subsample(self, data_dict, subsample_indices, is_tpose):
        sdf_list = []
        flow_list = []
        # normal_list = []

        # SDF samples
        if is_tpose and self.num_points_sdf > 0:
            points_sdf = data_dict['points_sdf'][0]

            ind = np.random.default_rng().choice(points_sdf.shape[0], self.num_points_sdf, replace=False)
            points_sdf = points_sdf[ind]

            assert points_sdf.shape[0] == self.num_points_sdf, f"{points_sdf.shape[0]} vs {self.num_points_sdf}"
            sdf_list.append(points_sdf)

        # Flow samples
        if not is_tpose and self.num_points_flow > 0:
            for idx in range(len(data_dict['points_flow'])):
                flow_sample_points = data_dict['points_flow'][idx]
                # flow_sample_normals = data_dict['points_normal'][idx]
                points_flow = flow_sample_points[subsample_indices]
                # points_normal = flow_sample_normals[subsample_indices]
                # print(idx, points_flow.shape, points_normal.shape)

                flow_list.append(np.array(points_flow, dtype=np.float32))
                # normal_list.append(np.array(points_normal, dtype=np.float32))
                assert len(points_flow) == self.num_points_flow, f"{len(points_flow)} vs {self.num_points_flow}"
                # assert len(points_normal) == self.num_points_flow, f"{len(points_normal)} vs {self.num_points_flow}"
            # print("subsample", len(flow_list), flow_list[5].shape)

        return {
            'points_sdf': sdf_list,
            'points_flow': flow_list,
            # 'points_normal': normal_list,
            'path': data_dict['path'],
            'identity_id': data_dict['identity_id']
        }

    def _get_identity_id(self, d):
        identity_id = d['identity_id']
        assert identity_id < self.num_identities, f"Identity {identity_id} is not defined in labels_tpose.json"
        return identity_id

    def __getitem__(self, idx):

        if self.cache_data:
            data_dict = {}
            data_ref_dict = self.cache_tpose[idx]
            if not self.only_shape:
                data_dict = self.cache_flow[idx]
        else:
            # Load samples
            data_dict = {}
            data_ref_dict = self._load_sample(idx, self.sample_paths[idx], is_tpose=True)
            if self.shape_ft:
                return {
                    'ref': data_ref_dict,
                    'cur': data_dict,
                    'idx': int(self.instance_filenames[idx].split('-')[0]),
                    'motion_name': os.path.basename(data_ref_dict['path']),
                }
            data_dict = self._load_sample(idx, self.sample_paths[idx], is_tpose=False)
            if self.pose_ft:
                return {
                    'ref': data_ref_dict,
                    'cur': data_dict,
                    'idx': int(self.instance_filenames[idx].split('-')[0]),
                    'motion_name': os.path.basename(data_ref_dict['path']),
                }

        # Sample random indices for each sequence
        subsample_indices = np.random.randint(0, 200000, self.num_points_flow)

        # Subsample
        data_ref_dict = self._subsample(data_ref_dict, subsample_indices, is_tpose=True)

        if not self.only_shape:
            data_dict = self._subsample(data_dict, subsample_indices, is_tpose=False)

        return {
            'ref': data_ref_dict,
            'cur': data_dict,
            'idx': self.indice_list[idx],
            'motion_name': os.path.basename(data_ref_dict['path']),
        }

    def get_loader(self, shuffle=True):

        assert self.batch_size <= len(self), f"batch size ({self.batch_size}) > len dataset ({len(self)})"

        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True,
            drop_last=True
        )

    def get_batch_size(self):
        return self.batch_size

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)