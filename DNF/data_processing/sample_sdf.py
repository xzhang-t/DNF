import os
import json
import numpy as np
import glob
import trimesh
import mesh2sdf
import random
import torch
import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    # parser.add_argument('-lst', type=str, required=True)
    args = parser.parse_args()

    mesh_scale = 0.5
    size = 256
    level = 2 / size

    data_dir = '/cluster/falas/xzhang/DT4D_test'
    # split_file_path = f'./DT4D/{args.lst}.lst'
    split_file_path = './DT4D/train.lst'
    split_lst = sorted([line.rstrip('\n') for line in open(split_file_path)])


    cnt = 0
    for i in range(0, len(split_lst)):
        cnt += 1
        output_dir = os.path.join(data_dir, split_lst[i], 'points_seq')
        os.makedirs(output_dir, exist_ok=True)
        check_path = os.path.join(output_dir, '00000000.npz')
        if os.path.isfile(check_path):
            print("file exist: ", cnt, check_path)
            continue
        print("processing ", cnt, split_lst[i])
        filename = os.path.join(data_dir, split_lst[i], 'norm_meshes_seq','t-pose_normalized.ply')
        output_path = check_path
        mesh = trimesh.load(filename, force='mesh', process=False, maintain_order=True)
        vertices = mesh.vertices

        sdf, mesh = mesh2sdf.compute(
            vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
        sdf = sdf.reshape(-1)
        N = size
        bbox_min = -1
        bbox_max = 1
        voxel_origin = [bbox_min] * 3
        voxel_size = (bbox_max - bbox_min) / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        vol_num = 50000
        ind = np.random.default_rng().choice(samples.shape[0], vol_num, replace=False)
        sdf_vol = torch.Tensor(sdf[ind])
        points_vol = samples[ind]

        index = np.where((sdf > -0.02) & (sdf < 0.02))
        surface_num = 150000
        sdf = torch.Tensor(sdf[index])
        samples = samples[index]
        if samples.shape[0] < surface_num:
            ind = np.random.default_rng().choice(samples.shape[0], surface_num, replace=True)
        else:
            ind = np.random.default_rng().choice(samples.shape[0], surface_num, replace=False)
        sdf = sdf[ind]
        points = samples[ind]
        sdf = torch.cat([sdf_vol, sdf], dim=0)
        points = torch.cat([points_vol, points], dim=0)
        points[:, 3] = sdf
        print(output_path)
        np.savez(output_path, points=points)
        sys.stdout.flush()

