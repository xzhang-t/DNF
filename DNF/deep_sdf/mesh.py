#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import trimesh
import torch
import mcubes

import deep_sdf.utils

def create_mesh(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    # voxel_origin = [-1, -1, -1]
    # voxel_size = 2.0 / (N - 1)
    bbox_min = -0.55
    bbox_max = 0.55
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

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    vertices, triangles = mcubes.marching_cubes(sdf_values.numpy(), 0)

    # Normalize vertices to be in [-1, 1]
    # step = (bbox_max - bbox_min) / (N - 1)
    # vertices = np.multiply(vertices, step)
    # vertices += [bbox_min, bbox_min, bbox_min]
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(filename + '.obj')

    end = time.time()
    print("sampling takes: %f" % (end - start))

    return mesh, sdf_values