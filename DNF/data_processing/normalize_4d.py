import os
import json
import numpy as np
import glob
import trimesh
import random
import torch

def anime_read(filename):
    f = open(filename, "rb")
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data

def normalize_mesh(mesh, b_max, b_min, extent):
    # Global normalization
    vertices = (mesh.vertices - (b_max + b_min) / 2) / extent * 1.8
    mesh.vertices = vertices
    bbox_bounds = mesh.bounds
    # print(bbox_bounds)
    return mesh

split_file_path = './DT4D/train.lst'
split_lst = sorted([line.rstrip('\n') for line in open(split_file_path)])
base_path = '/cluster/andram/xzhang/animals'
data_dir = '/cluster/falas/xzhang/DT4D'

for item in split_lst:
    print(item)
    animal_path = os.path.join(base_path, item)
    anime_file_path = os.path.join(animal_path, item + ".anime")
    nf, nv, nt, vert_data, face_data, offset_data = anime_read(anime_file_path)

    save_dir = os.path.join(data_dir, item)
    normalized_mesh_dir = os.path.join(save_dir, 'norm_meshes_seq')
    os.makedirs(normalized_mesh_dir, exist_ok=True)
    normalized_mesh_path = os.path.join(normalized_mesh_dir, 't-pose_normalized.ply')

    check_path = os.path.join(normalized_mesh_dir, '16-pose_normalized.ply')
    if os.path.isfile(check_path):
        print("file exist: ", check_path)
        continue

    vert_datas = []
    total_time = 16
    v_min, v_max = float("inf"), float("-inf")
    total_time = min(nf-1, total_time)
    print("total frame", total_time)
    selceted = []
    # for t in np.linspace(0, nf, total_time, dtype=int, endpoint=True):
    # for t in range(0, 9):
    for t in np.linspace(0, total_time, 17, dtype=int):
        selceted.append(t)
        vert_data_copy = vert_data - np.mean(vert_data, axis=0, keepdims=True)
        if t > 0:
            vert_data_copy = vert_data + offset_data[t - 1] - np.mean(vert_data, axis=0, keepdims=True)
        vert_datas.append(vert_data_copy)
    print("selceted frames", selceted)
    print(len(vert_datas))
    t_mesh = trimesh.Trimesh(vert_datas[0], face_data)

    bbox_bounds = t_mesh.bounds
    b_min = bbox_bounds[0]
    b_max = bbox_bounds[1]
    extent = np.max(b_max - b_min)

    t_mesh = normalize_mesh(t_mesh, b_max, b_min, extent)
    trimesh.Trimesh.export(t_mesh, normalized_mesh_path, 'ply')

    print("Writing meshes into:", normalized_mesh_dir)

    for idx in range(1, len(vert_datas)):
        obj = trimesh.Trimesh(vert_datas[idx], face_data)
        obj = normalize_mesh(obj, b_max, b_min, extent)
        normalized_tmesh_path = os.path.join(normalized_mesh_dir, str(idx) + "-pose_normalized.ply")
        trimesh.Trimesh.export(obj, normalized_tmesh_path, 'ply')