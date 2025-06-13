import glob
import os
import random

import numpy as np
import torch
import trimesh
import json
import sys
from tqdm import tqdm
from models.shape_decoder import ShapeDecoder
from models.pose_decoder import PoseDecoder
from models.dictionary import Dict_acc
from models.dictionary_pose import Dict_pose_acc
import utils.deepsdf_utils as deepsdf_utils
from evaluation_metrics_3d import compute_all_metrics, compute_all_metrics_4d
from transformer import Transformer
from hyperdiffusion import HyperDiffusion
import configs_eval.config_eval_shape as shape_cfg
import configs_eval.config_eval_pose as pose_cfg

def load_checkpoint(exp_name, checkpoint):
    exp_dir = '/cluster/falas/xzhang/experiments/animal'
    checkpoint_dir = os.path.join(exp_dir, exp_name, "checkpoints")

    if isinstance(checkpoint, int):
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{checkpoint}.tar")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint}_checkpoint.tar")

    if not os.path.exists(checkpoint_path):
        raise Exception(f'Other checkpoint {checkpoint_path} does not exist!')

    return torch.load(checkpoint_path)

def MLP_SVD(shape_decoder, K):
    torch.nn.utils.remove_weight_norm(shape_decoder.lin0)
    torch.nn.utils.remove_weight_norm(shape_decoder.lin1)
    torch.nn.utils.remove_weight_norm(shape_decoder.lin2)
    torch.nn.utils.remove_weight_norm(shape_decoder.lin3)
    torch.nn.utils.remove_weight_norm(shape_decoder.lin4)
    torch.nn.utils.remove_weight_norm(shape_decoder.lin5)
    torch.nn.utils.remove_weight_norm(shape_decoder.lin6)
    torch.nn.utils.remove_weight_norm(shape_decoder.lin7)

    state_dict = shape_decoder.state_dict()
    layers = []
    layer_names = []
    bias = []
    weights = []
    U = []
    Sigma_ori = []
    Vt = []

    for i, l in enumerate(state_dict):
        layer_names.append(l)
        if 'bias' in l:
            bias.append(state_dict[l])
        else:
            weights.append(state_dict[l])

    for i in range(len(weights)):
        w = weights[i]
        layers.append(w)
        u, sigma, vt = torch.linalg.svd(layers[i])
        U.append(u)
        Sigma_ori.append(torch.log(sigma))  # Sigma = ln(sigma) [fs,259]
        Vt.append(vt)

    for i in range(len(U)):
        k = min(K, Sigma_ori[i].shape[0])
        U[i] = U[i][:, :k]
        Sigma_ori[i] = Sigma_ori[i][:k]
        Vt[i] = Vt[i][:k, :]
        print("shape of U Sigma Vt, ", i, U[i].shape, Sigma_ori[i].shape, Vt[i].shape)
    return bias, U, Sigma_ori, Vt

def normalize_vertices(vertices, v_min, v_max):
    vertices -= np.mean(vertices, axis=0, keepdims=True)
    vertices *= 0.95 / (max(abs(v_min), abs(v_max)))
    return vertices

def calc_metrics_4d(n_points, sample_pcs):
    split_file = './animals/diff_test.lst'
    test_object_names = sorted([line.rstrip('\n') for line in open(split_file)])

    ref_pcs=[]
    for item in test_object_names:
        v_min, v_max = float("inf"), float("-inf")
        pc_per_anim = []
        pcs = []
        mesh_path = os.path.join('/cluster/andram/xzhang/DT4D_con', item, 'norm_meshes_seq')
        for i in range(0,16):
            if i == 0:
                mesh_dir = os.path.join(mesh_path, 't-pose_normalized.ply')
            else:
                mesh_dir = os.path.join(mesh_path, f'{i}-pose_normalized.ply')
            obj = trimesh.load(mesh_dir)
            pc = obj.sample(n_points)
            pcs.append(pc) # 8, 2048
            pc_zero_cent = pc - np.mean(pc, axis=0, keepdims=True)
            v_min = min(v_min, np.amin(pc_zero_cent))
            v_max = max(v_max, np.amax(pc_zero_cent))
        for i, pc in enumerate(pcs):
            pc = normalize_vertices(pc, v_min, v_max)
            pc_per_anim.append(pc)
        ref_pcs.append(pcs)
    ref_pcs = np.array(ref_pcs)
    print("ref pcs",ref_pcs.shape)

    metrics = compute_all_metrics_4d(
        torch.tensor(sample_pcs).float().cuda(),
        torch.tensor(ref_pcs).float().cuda(),
        160,
    )
    return metrics


shape_mlp_dim = [512, 512, 512, 77, 512, 512, 512, 512, 1]
pose_mlp_dim = [1024, 1024, 1024, 205, 1024, 1024, 1024, 1024, 3]
shape_k = 384
shape_rank = 256
pose_k = 768
pose_rank = 512
device = torch.device("cuda")
number_of_samples_to_generate = 20
save_mesh = True
n_points = 2048
calc_metrics = False

shape_with_ft = True
shape_layers = []
shape_layer_names = []
shape_layers.append(shape_cfg.shape_codes_dim)
shape_layer_names.append('shape_latent')

if shape_with_ft:
    shape_sigma_length = []
    for dim in shape_mlp_dim:
        shape_sigma_length.append(min(dim, shape_k) + shape_rank)
    for i, dim in enumerate(shape_sigma_length):
        shape_layers.append(dim)
        shape_layer_names.append('sigma_' + str(i))

pose_with_ft = True
pose_layers = []
pose_layer_names = []
pose_layers.append(pose_cfg.pose_codes_dim)
pose_layer_names.append('pose_latent')

if pose_with_ft:
    pose_sigma_length = []
    for dim in pose_mlp_dim:
        pose_sigma_length.append(min(dim, pose_k) + pose_rank)
    for i, dim in enumerate(pose_sigma_length):
        pose_layers.append(dim)
        pose_layer_names.append('sigma_' + str(i))


#### shape decoder #####
shape_codes_dim = shape_cfg.shape_codes_dim
shape_decoder = ShapeDecoder(shape_codes_dim, **shape_cfg.shape_network_specs).cuda()
checkpoint = load_checkpoint(shape_cfg.shape_dir, shape_cfg.checkpoint)
print("Loaded SHAPE decoder", shape_cfg.shape_dir)
shape_decoder.load_state_dict(checkpoint['model_state_dict_shape_decoder'])

######## shape ft decoder #########
ft_shape_decoder = Dict_acc(shape_rank, 0,
                      positional_enc=shape_cfg.shape_network_specs['positional_enc']).cuda().eval()
bias,U,Sigma,Vt =MLP_SVD(shape_decoder, shape_k)
ft_shape_decoder.init_para(bias,U,Vt)
checkpoint_ft = load_checkpoint(shape_cfg.ft_shape_dir, shape_cfg.ckpt)
print("Loaded SHAPE ft", shape_cfg.ft_shape_dir)
ft_shape_decoder.load_state_dict(checkpoint_ft['model_state_dict_ft_decoder'])
ft_shape_decoder = ft_shape_decoder.cuda().eval()

###### shape diffusion #####
shape_input_size = [1, sum(shape_layers)]  #[1, 384]
print("shape input_size", shape_input_size)
shape_model = Transformer(
            shape_layers, shape_layer_names, **shape_cfg.transformer_config).to(device)

shape_diffuser = HyperDiffusion(
        shape_model, shape_layers, shape_input_size, shape_cfg
    )
shape_state_dict = torch.load(shape_cfg.shape_ckpt_path)
print("Loaded shape diffusion ckpt from", shape_cfg.shape_ckpt_path)
shape_diffuser.load_state_dict(shape_state_dict['state_dict'])


###### pose decoder ######
pose_codes_dim = pose_cfg.pose_codes_dim
pose_decoder = PoseDecoder(shape_codes_dim + pose_codes_dim, **pose_cfg.pose_network_specs).cuda()
checkpoint_pose = load_checkpoint(pose_cfg.pose_dir, pose_cfg.ckpt)
print("Loaded POSE decoder", pose_cfg.pose_dir)
pose_decoder.load_state_dict(checkpoint_pose['model_state_dict_pose_decoder'])

######## pose ft decoder #########
ft_pose_decoder = Dict_pose_acc(pose_rank, 0,
                      positional_enc=pose_cfg.pose_network_specs['positional_enc']).cuda().eval()
bias,U,Sigma,Vt =MLP_SVD(pose_decoder, pose_k)
ft_pose_decoder.init_para(bias,U,Vt)
checkpoint_ft_pose = load_checkpoint(pose_cfg.ft_pose_dir, pose_cfg.ckpt)
print("Loaded POSE ft", pose_cfg.ft_pose_dir)
ft_pose_decoder = torch.nn.DataParallel(ft_pose_decoder)
ft_pose_decoder.load_state_dict(checkpoint_ft_pose['model_state_dict_ft_decoder'])
ft_pose_decoder = ft_pose_decoder.cuda().eval()

###### pose diffusion #####
pose_input_size = [1, pose_cfg.diff_frame, sum(pose_layers)]
print("pose input_size", pose_input_size)
pose_model = Transformer(
            pose_layers, pose_layer_names, **pose_cfg.transformer_config).to(device)

pose_diffuser = HyperDiffusion(
        pose_model, pose_layers, pose_input_size, pose_cfg
    )
# pose_ckpt_path = '/cluster/falas/xzhang/experiments/animal_diff/2024-11-02__NPMs__POSE__Diff__wFT__bs48__ON__data_split_diff/checkpoints/last-epoch=1341-train_loss=0.00.ckpt'
# pose_ckpt_path = '/cluster/falas/xzhang/experiments/animal_diff/2024-11-03__NPMs__POSE__Diff__wFT__bs48__ON__data_split_diff__with_T/checkpoints/last-epoch=862-train_loss=0.00.ckpt'
# pose_ckpt_path = '/cluster/falas/xzhang/experiments/animal_diff/2024-11-06__NPMs__POSE__Diff__wFT__bs64__ON__data_split_diff__with_T_Aug/checkpoints/last-epoch=1499-train_loss=0.00.ckpt'
# pose_ckpt_path = '/cluster/falas/xzhang/experiments/animal_diff/2024-11-09__NPMs__POSE__Diff__wFT__reverse__cond_j__bs72__ON__data_split_diff/checkpoints/last-epoch=999-train_loss=0.00.ckpt'
pose_state_dict = torch.load(pose_cfg.pose_ckpt_path)
print("Loaded pose diffusion ckpt from", pose_cfg.pose_ckpt_path)
pose_diffuser.load_state_dict(pose_state_dict['state_dict'])

#### generate shape para #######
sample_x_0s_shape = shape_diffuser.diff.ddim_sample_loop(
    shape_diffuser.model, (number_of_samples_to_generate, *shape_input_size[1:]), cond=None)

save_dir = './generation/diff/shape+pose_vis'
os.makedirs(save_dir, exist_ok=True)
torch.save(sample_x_0s_shape, os.path.join(save_dir, 'sample_x_0s_shape.pth'))

split_shape_parameters = torch.split(sample_x_0s_shape, shape_layers, dim=-1)
shape_sigma_all = []
for cnt, para in enumerate(split_shape_parameters):
    if cnt == 0:
        shape_codes = split_shape_parameters[cnt]
    else:
        tmp = split_shape_parameters[cnt]
        shape_sigma_all.append(tmp)

#### generate pose para #######
cond_in = shape_codes.unsqueeze(1)
input_shape = (number_of_samples_to_generate, *pose_input_size[1:])
sample_x_0s_pose = pose_diffuser.diff.ddim_sample_loop(
                    pose_diffuser.model, input_shape, cond_in)
print("sample_x_0s_pose", sample_x_0s_pose.shape)
torch.save(sample_x_0s_pose, os.path.join(save_dir, 'sample_x_0s_pose.pth'))

frames_gen = sample_x_0s_pose

all_frame = 18
gen_frame = 4

sample_num = int((all_frame - pose_cfg.diff_frame) / gen_frame) #6
for num in range(sample_num):
    con_frames = pose_cfg.diff_frame - gen_frame
    input_frames = frames_gen[:, -con_frames:, :] # last few frames as condition
    # print("input_frames", input_frames.shape)  # [bs, n, 384]

    input_noise = torch.randn(input_frames.shape[0],gen_frame,input_frames.shape[2]).cuda()
    # print("noise", input_noise.shape)
    input_noise = torch.cat((input_frames,input_noise),dim=1)

    sample_x_0s_pose = pose_diffuser.diff.ddim_sample_loop(
        pose_diffuser.model, input_shape,
        cond=cond_in, noise=input_noise, inpaint=True, gen_frame=gen_frame)

    frames_gen = torch.cat((frames_gen, sample_x_0s_pose[:, -gen_frame:, :]), dim=1)
    # print('frames generated', frames_gen.shape)  #[16, 8, 10576]

torch.save(frames_gen, os.path.join(save_dir, 'frames_gen.pth'))

split_pose_parameters = torch.split(frames_gen, pose_layers, dim=-1)
pose_sigma_all = []
for cnt, para in enumerate(split_pose_parameters):
    if cnt == 0:
        pose_codes = split_pose_parameters[cnt]
    else:
        tmp = split_pose_parameters[cnt]
        pose_sigma_all.append(tmp)

sample_pcs=[]
for idx in range(number_of_samples_to_generate):
    print(f"processing {idx}-th item")
    v_min, v_max = float("inf"), float("-inf")
    shape_code = shape_codes[idx].unsqueeze(0).cuda() #1,384
    # print("shape code", shape_code.shape)

    Sigma_s = torch.zeros(len(shape_sigma_all), shape_sigma_all[1].shape[-1])
    # print("Sigma_s shape", Sigma_s.shape)  # 9, 640
    for layer, sigma in enumerate(shape_sigma_all):
        # print('sigma', sigma.shape) # bs, dim
        Sigma_s[layer, :sigma.shape[1]] = sigma[idx]
    sigma_shape = Sigma_s.unsqueeze(0).cuda()

    t_mesh = deepsdf_utils.create_mesh_from_code(
        ft_shape_decoder, shape_code.unsqueeze(0), sigma=sigma_shape, bs=1, shape_codes_dim=shape_codes_dim,
        N=256, max_batch=int(2 ** 18))
    if save_mesh:
        save_path = os.path.join(save_dir, str(idx))
        os.makedirs(save_path, exist_ok=True)
        t_mesh.export(save_path + "/t_ft.obj")

    p_ref = t_mesh.vertices.astype(np.float32)
    points = torch.from_numpy(p_ref)[None, :].cuda()
    points_flat = points.reshape(-1, 3)

    pose_code_n = pose_codes[idx]

    pcs = []
    for f_num in range(pose_code_n.shape[0]):
        shape_codes_inputs = shape_code.expand(points_flat.shape[0], -1)
        # print("shape_codes_shape", shape_codes_inputs.shape)

        pose_code = pose_code_n[f_num].unsqueeze(0)  # 1,384
        pose_codes_inputs = pose_code.expand(points_flat.shape[0], -1)
        # print("pose_codes_shape", pose_codes_inputs.shape)

        shape_pose_codes_inputs = torch.cat([shape_codes_inputs, pose_codes_inputs], 1)
        # print("shape + pose", shape_pose_codes_inputs.shape)

        pose_inputs = torch.cat([shape_pose_codes_inputs, points_flat], 1)
        # print("pose input", pose_inputs.shape)

        Sigma_p = torch.zeros(len(pose_sigma_all), pose_sigma_all[1].shape[-1]) # bs, 8, 1280
        # print("Sigma_p shape", Sigma_p.shape)  # 9,1280
        for layer, sigma in enumerate(pose_sigma_all):
            # print('sigma', sigma.shape) # bs, 8, 1280
            Sigma_p[layer, :sigma.shape[-1]] = sigma[idx][f_num]
        sigma_pose = Sigma_p.unsqueeze(0).cuda() # 1,9,1280

        p_ref_warped = ft_pose_decoder(pose_inputs, sigma_pose, 1)
        p_ref_warped = p_ref_warped.detach().cpu().numpy()

        flow_mesh = t_mesh.copy()
        flow_mesh.vertices = p_ref_warped

        if save_mesh:
            print("export flow mesh", idx, f_num)
            flow_mesh.export(save_path + f"/{f_num}_ft.obj")

        pc = flow_mesh.sample(n_points)
        pcs.append(pc)
        pc_zero_cent = pc - np.mean(pc, axis=0, keepdims=True)
        v_min = min(v_min, np.amin(pc_zero_cent))
        v_max = max(v_max, np.amax(pc_zero_cent))
    pc_per_anim = []
    for j, pc in enumerate(pcs):
        pc = normalize_vertices(pc, v_min, v_max)
        pc_per_anim.append(pc)
    sample_pcs.append(pc_per_anim)
sample_pcs = np.array(sample_pcs)
print("sample pcs",sample_pcs.shape)

if calc_metrics:
    metrics = calc_metrics_4d(n_points, sample_pcs)
    for metric_name in metrics:
        print("test/" + metric_name, metrics[metric_name])





