#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data


# def get_instance_filenames_npy(data_source, split):
#     npyfiles = []
#     for dataset in split:
#         for class_name in split[dataset]:
#             for idx,instance_name in enumerate(split[dataset][class_name]):
#                 instance_filename = os.path.join(
#                     dataset, class_name, instance_name
#                 )
#                 if not os.path.isfile(
#                     os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
#                 ):
#                     logging.warning(
#                         "Requested non-existent file '{}'".format(instance_filename)
#                     )
#                 else:
#                     npyfiles += [str(idx)+'_'+instance_filename]
#     return npyfiles


# def get_instance_filenames(data_source, split):
#     npzfiles = []
#     print(os.path.join(data_source, ws.sdf_samples_subdir))
#     for dataset in split:
#         for class_name in split[dataset]:
#             for idx, instance_name in enumerate(split[dataset][class_name]):
#                 instance_filename = os.path.join(
#                     dataset, class_name, instance_name + ".npz"
#                 )
#                 if not os.path.isfile(
#                     os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
#                 ):
#                     # raise RuntimeError(
#                     #     'Requested non-existent file "' + instance_filename + "'"
#                     # )
#                     logging.warning(
#                         "Requested non-existent file '{}'".format(instance_filename)
#                     )
#                 else:
#                     # test = os.path.join('./examples/planes/ft_dic_s_adap_multi', instance_name + '.pth')
#                     # if not os.path.isfile(test):
#                     npzfiles += [str(idx)+'_'+instance_filename]
#     print("!!!!!!length of training samples", len(npzfiles))
#     return npzfiles


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


# class LatentDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         latent_folder,
#         num_scenes,
#         latent_size,
#         code_bound,
#         filename
#     ):
#         self.latent_folder = latent_folder
#         self.num_scenes = num_scenes
#         self.latent_size = latent_size
#         self.filename = filename + ".pth"
#         self.lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
#
#         full_filename = os.path.join(
#             ws.get_latent_codes_dir(latent_folder), self.filename
#         )
#
#         if not os.path.isfile(full_filename):
#             raise Exception('latent state file "{}" does not exist'.format(full_filename))
#         print("Loading from: ", full_filename)
#         data = torch.load(full_filename)
#
#         self.lat_vecs.load_state_dict(data["latent_codes"])
#         print("training latent_codes: ",self.lat_vecs)
#
#     def __len__(self):
#         return self.lat_vecs.num_embeddings
#
#     def __getitem__(self, idx):
#         # print("get item: ",torch.tensor(idx))
#         vec = self.lat_vecs(torch.tensor(idx)).detach()
#         # print(vec.shape)
#         return vec, idx


def get_instance_sigma(data_source, split):
    sigmafiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for idx, instance_name in enumerate(split[dataset][class_name]):
                instance_filename = str(idx)+'_'+instance_name + ".pth"
                # instance_filename = instance_name + ".pth"
                if not os.path.isfile(
                    os.path.join(data_source, instance_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                else:
                    sigmafiles += [instance_filename]

    return sigmafiles


def get_instance_sigma_animals(data_source, split):
    sigmafiles = []
    for idx, instance_name in enumerate(split['train_set']):
        # instance_filename = str(idx)+'_'+instance_name + ".pth"
        instance_filename = instance_name + ".pth"
        if not os.path.isfile(
            os.path.join(data_source, instance_filename)
        ):
            logging.warning(
                "Requested non-existent file '{}'".format(instance_filename)
            )
        else:
            sigmafiles += [instance_filename]
    return sigmafiles


class DicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        latent_folder,
        latent_size,
        code_bound,
        filename,
    ):
        self.data_source = data_source
        self.sigma_file = get_instance_sigma_animals(data_source, split)
        # print(self.sigma_file)
        self.num_scenes = len(self.sigma_file)
        self.latent_size = latent_size
        self.filename = filename + ".pth"
        self.u_file = os.path.join(data_source, 'U.pth')
        self.U = torch.load(self.u_file, map_location='cpu')
        self.v_file = os.path.join(data_source, 'Vt.pth')
        self.Vt = torch.load(self.v_file, map_location='cpu')

        # self.vt_file = os.path.join(data_source, 'Vt.pth')

        # self.lat_vecs = torch.nn.Embedding(self.num_scenes, latent_size, max_norm=code_bound)

        # full_filename = os.path.join(latent_folder, f"checkpoint_epoch_{filename}.tar")
        full_filename = os.path.join(latent_folder, f"latest_checkpoint.tar")

        if not os.path.isfile(full_filename):
            raise Exception('latent state file "{}" does not exist'.format(full_filename))
        print("Loading from: ", full_filename)
        data = torch.load(full_filename)

        self.lat_vecs = data['shape_codes'].detach().clone()
        # self.lat_vecs.load_state_dict(data["latent_codes"])
        # print("training latent_codes: ", self.lat_vecs)

    def __len__(self):
        return len(self.sigma_file)

    def __getitem__(self, idx):
        weights = []
        # vec = self.lat_vecs(torch.tensor(idx)).detach()
        vec = self.lat_vecs[idx, ...].reshape(-1)
        # print("!!!!",vec.shape)
        weights.append(vec)

        path = self.sigma_file[idx]
        # print(idx,path)
        sigma = torch.load(os.path.join(self.data_source, path),map_location='cpu')
        # print("load sigma")

        for i in range(len(self.Vt)):
            tmp = torch.zeros(self.Vt[i].shape[0] - sigma[i].shape[0])
            sigma[i] = torch.cat((torch.exp(sigma[i]), tmp), 0)
            # tmp = torch.full((U[i].shape[1] - sigma[i].shape[0],), -100.0)
            # sigma[i] = torch.cat((torch.clamp(sigma[i],min=-100.0), tmp), 0)
            weights.append(sigma[i])
        weights = torch.hstack(weights)
        # print(weights.shape)

        return weights,idx





