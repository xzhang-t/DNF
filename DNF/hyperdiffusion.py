import copy
import glob
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh
import json
import sys
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from tqdm import tqdm
from models.shape_decoder import ShapeDecoder
from models.pose_decoder import PoseDecoder
from models.dictionary import Dict_acc
from models.dictionary_pose import Dict_pose_acc
import utils.deepsdf_utils as deepsdf_utils

from diffusion.gaussian_diffusion import (GaussianDiffusion, LossType,
                                          ModelMeanType, ModelVarType)
from evaluation_metrics_3d import compute_all_metrics, compute_all_metrics_4d

class HyperDiffusion(pl.LightningModule):
    def __init__(
        self, model, layers, image_shape, cfg
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.only_shape = cfg.only_shape
        self.layers = layers
        fake_data = torch.randn(*image_shape)

        encoded_outs = fake_data
        print("encoded_outs.shape", encoded_outs.shape)  #[48, 4, 384]
        timesteps = cfg.timesteps
        betas = torch.tensor(np.linspace(1e-4, 2e-2, timesteps))
        self.image_size = encoded_outs[:1].shape
        print("image size",self.image_size) # [1,4,384]

        # Initialize diffusion utiities
        self.diff = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType[cfg.diff_config['model_mean_type']],
            model_var_type=ModelVarType[cfg.diff_config['model_var_type']],
            loss_type=LossType[cfg.diff_config['loss_type']],
            diff_pl_module=self,
        )

    def forward(self, images, cond):
        # print("forward!!!!!!!")
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(images.shape[0],))
            .long()
            .to(self.device)
        )
        images = images * self.cfg.normalization_factor
        x_t, e = self.diff.q_sample(images, t)
        x_t = x_t.float()
        e = e.float()
        return self.model(x_t, t, cond), e

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        if self.cfg.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.cfg.scheduler_step, gamma=0.9
            )
            return [optimizer], [scheduler]
        return optimizer

    def grid_to_mesh(self, grid):
        grid = np.where(grid > 0, True, False)
        vox_grid = trimesh.voxel.VoxelGrid(grid)
        try:
            vox_grid = vox_grid.marching_cubes
        except:
            return vox_grid.as_boxes()
        vert = vox_grid.vertices
        if len(vert) == 0:
            return vox_grid
        vert /= grid.shape[-1]
        vert = 2 * vert - 1
        vox_grid.vertices = vert
        return vox_grid

    def training_step(self, train_batch, batch_idx):
        # Extract input_data (either voxel or weight) which is the first element of the tuple
        if not self.only_shape:
            cond = train_batch[0]
            input_data = train_batch[1]
        else:
            cond = None
            input_data = train_batch[0]

        # Sample a diffusion timestep
        # print("start training!!!!!!!")
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(input_data.shape[0],))
            .long()
            .to(self.device)
        )
        if self.cfg.cond_jitter:
            c_t = random.randint(0,25)
            input_t = torch.tensor([c_t] * cond.shape[0], device=self.device)
            cond,_ = self.diff.q_sample(cond, input_t)

        # Execute a diffusion forward pass
        loss_terms = self.diff.training_losses(
            self.model,
            input_data * self.cfg.normalization_factor,
            t,
            cond,
            self.logger,
            model_kwargs=None,
        )
        loss_mse = loss_terms["loss"].mean()
        self.log("train_loss", loss_mse)

        loss = loss_mse
        return loss

    def validation_step(self, val_batch, batch_idx):
        return

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        epoch_loss = sum(output["loss"] for output in outputs) / len(outputs)
        self.log("epoch_loss", epoch_loss)

    def load_checkpoint(self, exp_name, checkpoint):
        exp_dir = './experiments/animal'
        checkpoint_dir = os.path.join(exp_dir, exp_name, "checkpoints")

        if isinstance(checkpoint, int):
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{checkpoint}.tar")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint}_checkpoint.tar")

        if not os.path.exists(checkpoint_path):
            raise Exception(f'Other checkpoint {checkpoint_path} does not exist!')

        return torch.load(checkpoint_path)

    def MLP_SVD(self, pose_decoder, K):
        torch.nn.utils.remove_weight_norm(pose_decoder.lin0)
        torch.nn.utils.remove_weight_norm(pose_decoder.lin1)
        torch.nn.utils.remove_weight_norm(pose_decoder.lin2)
        torch.nn.utils.remove_weight_norm(pose_decoder.lin3)
        torch.nn.utils.remove_weight_norm(pose_decoder.lin4)
        torch.nn.utils.remove_weight_norm(pose_decoder.lin5)
        torch.nn.utils.remove_weight_norm(pose_decoder.lin6)
        torch.nn.utils.remove_weight_norm(pose_decoder.lin7)

        state_dict = pose_decoder.state_dict()
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

        # U, Sigma, Vt =  [], [], []
        for i in range(len(weights)):
            w = weights[i]
            layers.append(w)
            u, sigma, vt = torch.linalg.svd(layers[i])
            U.append(u)
            Sigma_ori.append(torch.log(sigma))  # Sigma = ln(sigma)
            Vt.append(vt)

        for i in range(len(U)):
            if self.cfg.compress:
                k = min(K, Sigma_ori[i].shape[0])
                U[i] = U[i][:, :k]
                Sigma_ori[i] = Sigma_ori[i][:k]
                Vt[i] = Vt[i][:k, :]
                print("shape of U Sigma Vt, ", i, U[i].shape, Sigma_ori[i].shape, Vt[i].shape)
            else:
                if U[i].shape[1] > Vt[i].shape[0]:
                    U[i] = U[i][:, :Vt[i].shape[0]]
                elif U[i].shape[1] < Vt[i].shape[0]:
                    Vt[i] = Vt[i][:U[i].shape[0]]
                print("shape of U Sigma Vt, ", i, U[i].shape, Sigma_ori[i].shape, Vt[i].shape)
        return bias, U, Sigma_ori, Vt

    def calc_metrics_4d(self, cond, shape_sigma=None):
        n_points = 2048
        split_file = './animals/diff_test.lst'
        test_object_names = sorted([line.rstrip('\n') for line in open(split_file)])

        ref_pcs=[]
        for item in test_object_names:
            pcs = []
            mesh_path = os.path.join('/cluster/andram/xzhang/DT4D_con', item, 'norm_meshes_seq')
            for i in range(0,16):
                if i == 0:
                    mesh_dir = os.path.join(mesh_path, 't-pose_normalized.ply')
                else:
                    mesh_dir = os.path.join(mesh_path, f'{i}-pose_normalized.ply')
                obj = trimesh.load(mesh_dir)
                pc = obj.sample(n_points)
                pcs.append(pc) # n, 2048
            ref_pcs.append(pcs)
        ref_pcs = np.array(ref_pcs)
        print("ref pcs",ref_pcs.shape)

        num_mesh_to_generates = len(ref_pcs)
        print("test number", num_mesh_to_generates)

        all_frame = 16
        gen_frame = 1
        base_dir = f'./generation/diff/pose_ft_train-{all_frame}-{gen_frame}'
        os.makedirs(base_dir, exist_ok=True)

        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        print("cond", cond.shape)
        cond_in = cond[:num_mesh_to_generates]
        input_shape = (num_mesh_to_generates, *self.image_size[1:])

        if os.path.exists(os.path.join(base_dir, 'frames_gen.pth')):
            print('load gen frames!')
            frames_gen = torch.load(os.path.join(base_dir, 'frames_gen.pth'))

        else:
            sample_x_0s = self.diff.ddim_sample_loop(
                self.model, input_shape, cond_in)
            print('sample_x_0s', sample_x_0s.shape)

            frames_gen = sample_x_0s

            sample_num = int((all_frame - self.cfg.diff_frame) / gen_frame)  # 6
            for num in range(sample_num):
                con_frames = self.cfg.diff_frame - gen_frame  # 4-1=3
                input_frames = sample_x_0s[:, -con_frames:, :]  # last 3 frames as condition
                print("input_frames", input_frames.shape)  # [bs, 2, 384]
                input_noise = torch.randn(input_frames.shape[0], gen_frame, input_frames.shape[2]).cuda()
                print("noise", input_noise.shape)
                input_noise = torch.cat((input_frames, input_noise), dim=1)

                sample_x_0s = self.diff.ddim_sample_loop(
                    self.model, input_shape,
                    cond=cond_in, noise=input_noise, inpaint=True, gen_frame=gen_frame)
                # print('sample_x_0s', sample_x_0s.shape)  # torch.Size([2, 4, 384])
                # print(sample_x_0s)

                frames_gen = torch.cat((frames_gen, sample_x_0s[:, -gen_frame:, :]), dim=1)
                print('frames generated', frames_gen.shape)

            torch.save(frames_gen, os.path.join(base_dir, 'frames_gen.pth'))

        sample_pcs = self.sample_pc_with_gen(base_dir, num_mesh_to_generates)

        metrics = compute_all_metrics_4d(
            torch.tensor(sample_pcs).float().to(self.device),
            torch.tensor(ref_pcs).float().to(self.device),
            160,
            self.logger,
        )

        return metrics

    def gen_mesh_with_pose_ft(self, cond_in, frames_gen, base_dir, n_points, shape_sigma=None):
        ft_shape_decoder = Dict_acc(256, 0, positional_enc=True).cuda().eval()
        bias, U, Sigma, Vt = self.MLP_SVD(self.shape_decoder, 384)
        ft_shape_decoder.init_para(bias, U, Vt)
        checkpoint_ft = self.load_checkpoint(self.cfg.ft_shape_dir, self.cfg.checkpoint)
        ft_shape_decoder.load_state_dict(checkpoint_ft['model_state_dict_ft_decoder'])
        ft_shape_decoder = ft_shape_decoder.cuda()

        ft_decoder = Dict_pose_acc(self.cfg.rank, 0, positional_enc=True).cuda().eval()
        bias, U, Sigma, Vt = self.MLP_SVD(self.pose_decoder, self.cfg.k)
        ft_decoder.init_para(bias, U, Vt)
        checkpoint_ft = self.load_checkpoint(self.cfg.ft_pose_dir, self.cfg.checkpoint)
        ft_decoder = torch.nn.DataParallel(ft_decoder)
        ft_decoder.load_state_dict(checkpoint_ft['model_state_dict_ft_decoder'])
        ft_decoder = ft_decoder.cuda()

        number_of_samples_to_generate = cond_in.shape[0]
        print('number_of_samples_to_generate',number_of_samples_to_generate)

        sample_pcs=[]
        for idx in range(number_of_samples_to_generate):
            save_path = os.path.join(base_dir, str(idx))
            os.makedirs(save_path, exist_ok=True)
            shape_code = cond_in[idx]  # 1,384

            if shape_sigma is not None:
                Sigma_s = torch.zeros(len(shape_sigma), shape_sigma[1].shape[1])
                print("Sigma_s shape", Sigma_s.shape)  # 9,1024
                for layer, sigma in enumerate(shape_sigma):
                    print('sigma', sigma.shape)
                    Sigma_s[layer, :sigma.shape[1]] = sigma[idx]
                # print(Sigma_s.shape)
                sigma_s_t = Sigma_s.unsqueeze(0).cuda()

                t_mesh = deepsdf_utils.create_mesh_from_code(
                    ft_shape_decoder, shape_code.unsqueeze(0), sigma=sigma_s_t, bs=1, shape_codes_dim=shape_code.shape[1],
                    N=256, max_batch=int(2 ** 18))
                t_mesh.export(save_path + "/t_ft.obj")
            else:
                t_mesh = deepsdf_utils.create_mesh_from_code(
                    self.shape_decoder, shape_code.unsqueeze(0), shape_codes_dim=self.cfg.shape_codes_dim, N=256,
                    max_batch=int(2 ** 18))
                t_mesh.export(save_path + "/t.obj")
            p_ref = t_mesh.vertices.astype(np.float32)
            points = torch.from_numpy(p_ref)[None, :].cuda()
            points_flat = points.reshape(-1, 3)

            x_0s = frames_gen[idx]  # [4, 10576]

            split_parameters = torch.split(x_0s, self.layers, dim=1)
            sigma_all = []
            for cnt, para in enumerate(split_parameters):
                if cnt == 0:
                    pose_codes = split_parameters[cnt]  # [4, 384]
                else:
                    tmp = split_parameters[cnt]  # [4,1024], [4,1024]
                    sigma_all.append(tmp)
            print("pose code",pose_codes.shape)
            print("sigma length", len(sigma_all))  # [[4,],[],[],[]]

            pcs=[]
            for f_num in range(pose_codes.shape[0]):
                shape_codes_inputs = shape_code.expand(points_flat.shape[0], -1)
                # print("shape_codes_shape", shape_codes_inputs.shape)

                pose_code = pose_codes[f_num].unsqueeze(0)  # 1,384
                pose_codes_inputs = pose_code.expand(points_flat.shape[0], -1)
                # print("pose_codes_shape", pose_codes_inputs.shape)

                shape_pose_codes_inputs = torch.cat([shape_codes_inputs, pose_codes_inputs], 1)

                pose_inputs = torch.cat([shape_pose_codes_inputs, points_flat], 1)

                Sigma = torch.zeros(len(sigma_all), sigma_all[1].shape[1])
                print("Sigma shape", Sigma.shape)  # 9,1024
                for layer, sigma in enumerate(sigma_all):
                    print('sigma', sigma.shape)
                    Sigma[layer, :sigma.shape[1]] = sigma[f_num]

                sigma_t = Sigma.unsqueeze(0)
                p_ref_warped = ft_decoder(pose_inputs, sigma_t, 1)

                p_ref_warped = p_ref_warped.detach().cpu().numpy()

                flow_mesh = t_mesh.copy()
                flow_mesh.vertices = p_ref_warped

                print("export flow mesh", f_num)
                flow_mesh.export(save_path + f"/{f_num}.obj")

                pc = flow_mesh.sample(n_points)
                pcs.append(pc)

                sys.stdout.flush()
            sample_pcs.append(pcs)
        sample_pcs = np.array(sample_pcs)
        print("sample pcs",sample_pcs.shape)

        return sample_pcs

    def sample_pc_with_gen(self,base_dir,num_mesh_to_generates):
        n_points = 2048
        sample_pcs = []
        for idx in range(num_mesh_to_generates):
            save_path = os.path.join(base_dir, str(idx))
            print('save path',save_path)
            pcs = []
            for i in range(0, 16):
                mesh_dir = os.path.join(save_path, f'{i}.obj')
                obj = trimesh.load(mesh_dir)
                pc = obj.sample(n_points)
                pcs.append(pc)
            sample_pcs.append(pcs)
        sample_pcs=np.array(sample_pcs)
        print("sample pcs", sample_pcs.shape)
        return sample_pcs

    def test_step(self, test_batch, batch_idx):
        print("testing!!!!!!")
        shape_codes_dim = self.cfg.shape_codes_dim
        self.shape_decoder = ShapeDecoder(shape_codes_dim, **self.cfg.shape_network_specs).cuda()
        checkpoint = self.load_checkpoint(self.cfg.shape_dir, self.cfg.checkpoint)
        print("Loaded SHAPE decoder", self.cfg.shape_dir)
        self.shape_decoder.load_state_dict(checkpoint['model_state_dict_shape_decoder'])

        if self.cfg.only_shape:
            cond = None
            input_data = test_batch[0]

            number_of_samples_to_generate = 4
            sample_x_0s = self.diff.ddim_sample_loop(
                self.model, (number_of_samples_to_generate, *self.image_size[1:]), cond=None)
            # print('sample_x_0s', sample_x_0s.shape)  # torch.Size([bs,384])  /  [bs,384+sigma]

            if self.cfg.with_ft:
                ft_decoder = Dict_acc(self.cfg.rank, 0, positional_enc=True).cuda().eval()
                bias, U, Sigma, Vt = self.MLP_SVD(self.shape_decoder, self.cfg.k)
                ft_decoder.init_para(bias, U, Vt)
                checkpoint_ft = self.load_checkpoint(self.cfg.ft_shape_dir, self.cfg.checkpoint)
                ft_decoder.load_state_dict(checkpoint_ft['model_state_dict_ft_decoder'])
                ft_decoder = ft_decoder.cuda()

                save_path = './generation/diff/shape_ft'
                os.makedirs(save_path, exist_ok=True)
                torch.save(sample_x_0s,os.path.join(save_path, 'sample_x_0s.pth'))
                if self.cfg.gen_mesh:
                    for idx in range(number_of_samples_to_generate):
                        x_0s = sample_x_0s[idx]  # [4, 10576]
                        split_parameters = torch.split(x_0s, self.layers, dim=-1)
                        sigma_all = []
                        for cnt, para in enumerate(split_parameters):
                            if cnt == 0:
                                shape_code = split_parameters[cnt].unsqueeze(0).cuda()  # [1, 384]
                            else:
                                tmp = split_parameters[cnt]  # [1024], [1024]
                                sigma_all.append(tmp)
                        # print("sigma length", len(sigma_all))  # [[1024],[1024],[],[]]

                        Sigma = torch.zeros(len(sigma_all), sigma_all[1].shape[0])
                        # print("Sigma shape", Sigma.shape)  # 9,1024
                        for layer, sigma in enumerate(sigma_all):
                            # print('sigma', sigma.shape)
                            Sigma[layer, :sigma.shape[0]] = sigma
                        sigma_t = Sigma.unsqueeze(0).cuda()

                        ft_mesh = deepsdf_utils.create_mesh_from_code(
                            ft_decoder, shape_code.unsqueeze(0), sigma=sigma_t, bs=1, shape_codes_dim=shape_codes_dim,
                            N=256, max_batch=int(2 ** 18))
                        ft_mesh.export(save_path + f"/{idx}_ft.obj")
                        sys.stdout.flush()

            else:
                save_path = './generation/diff/shape'
                os.makedirs(save_path, exist_ok=True)
                for idx in range(number_of_samples_to_generate):
                    shape_code = sample_x_0s[idx].unsqueeze(0)  # [4,384]
                    print("generated codes", shape_code.shape, shape_code)

                    t_mesh = deepsdf_utils.create_mesh_from_code(
                        self.shape_decoder, shape_code.unsqueeze(0), shape_codes_dim=shape_codes_dim, N=256,
                        max_batch=int(2 ** 18))
                    t_mesh.export(save_path + f"/{idx}_t.obj")
                    sys.stdout.flush()
        else:
            pose_codes_dim = self.cfg.pose_codes_dim
            self.pose_decoder = PoseDecoder(shape_codes_dim + pose_codes_dim, **self.cfg.pose_network_specs).cuda()
            checkpoint_pose = self.load_checkpoint(self.cfg.pose_dir, self.cfg.checkpoint)
            print("Loaded POSE decoder", self.cfg.pose_dir)
            self.pose_decoder.load_state_dict(checkpoint_pose['model_state_dict_pose_decoder'])

            if self.cfg.calc_metrics:
                if self.cfg.load_cond:
                    gens = torch.load(self.cfg.gen_path)
                    split_parameters = torch.split(gens, self.cfg.shape_layers, dim=-1)
                    cond = split_parameters[0]
                    metrics = self.calc_metrics_4d(cond=cond, shape_sigma=split_parameters[1:])
                else:
                    metrics = self.calc_metrics_4d(cond = test_batch[0])
                for metric_name in metrics:
                    print("test/" + metric_name, metrics[metric_name])
                return

            number_of_samples_to_generate = 8
            if self.cfg.load_cond:
                gens = torch.load(self.cfg.gen_path)
                split_parameters = torch.split(gens, self.cfg.shape_layers, dim=-1)
                shape_idx = [34, 63, 71, 72]
                cond = split_parameters[0] # [100, 384]
                cond_in = cond[shape_idx].unsqueeze(1)
                print("cond in", cond_in.shape)
            else:
                cond = test_batch[0]
                if cond.dim() == 2:
                    cond = cond.unsqueeze(1)
                print("cond", cond.shape)
                input_data = test_batch[1]
                if cond.shape[0]<number_of_samples_to_generate:
                    cond_in = cond.expand(number_of_samples_to_generate,-1,-1)
                else:
                    cond_in = cond[:number_of_samples_to_generate]

            input_shape = (number_of_samples_to_generate, *self.image_size[1:]) # torch.Size([2, 4, 384])
            sample_x_0s = self.diff.ddim_sample_loop(
                    self.model, input_shape, cond_in)
            print('sample_x_0s',sample_x_0s.shape)  # torch.Size([2, 4, 384])
            # print(sample_x_0s)

            frames_gen = sample_x_0s

            all_frame = 18
            gen_frame = 4
            sample_num = int((all_frame - self.cfg.diff_frame) / gen_frame) #6
            for num in range(sample_num):
                con_frames = self.cfg.diff_frame - gen_frame # 4-1=3
                input_frames = sample_x_0s[:, -con_frames:, :] # last 3 frames as condition
                input_noise = torch.randn(input_frames.shape[0],gen_frame,input_frames.shape[2]).cuda()
                input_noise = torch.cat((input_frames,input_noise),dim=1)

                sample_x_0s = self.diff.ddim_sample_loop(
                    self.model, input_shape,
                    cond=cond_in, noise=input_noise, inpaint=True, gen_frame=gen_frame)

                frames_gen = torch.cat((frames_gen, sample_x_0s[:, -gen_frame:, :]), dim=1)
                print('frames generated', frames_gen.shape)
                # print(frames_gen)
                sys.stdout.flush()

            if self.cfg.with_ft:
                base_dir = f'./generation/diff/pose_ft_16f_test-{all_frame}-{gen_frame}'
                os.makedirs(base_dir, exist_ok=True)
                torch.save(frames_gen, os.path.join(base_dir, 'frames_gen.pth'))

                ft_decoder = Dict_pose_acc(self.cfg.rank, 0, positional_enc=True).cuda()
                bias, U, Sigma, Vt = self.MLP_SVD(self.pose_decoder, self.cfg.k)
                ft_decoder.init_para(bias, U, Vt)
                checkpoint_ft = self.load_checkpoint(self.cfg.ft_pose_dir, self.cfg.checkpoint)
                ft_decoder = torch.nn.DataParallel(ft_decoder)
                ft_decoder.load_state_dict(checkpoint_ft['model_state_dict_ft_decoder'])
                ft_decoder = ft_decoder.cuda()

                for idx in range(number_of_samples_to_generate):
                    save_path = os.path.join(base_dir, str(idx))
                    os.makedirs(save_path, exist_ok=True)
                    shape_code = cond_in[idx]  # 1,384
                    t_mesh = deepsdf_utils.create_mesh_from_code(
                        self.shape_decoder, shape_code.unsqueeze(0), shape_codes_dim=shape_codes_dim, N=128,
                        max_batch=int(2 ** 18))
                    t_mesh.export(save_path + "/t.obj")

                    p_ref = t_mesh.vertices.astype(np.float32)
                    points = torch.from_numpy(p_ref)[None, :].cuda()
                    points_flat = points.reshape(-1, 3)

                    x_0s = frames_gen[idx]  #[4, 10576]

                    split_parameters = torch.split(x_0s, self.layers, dim=1)
                    sigma_all = []
                    for cnt, para in enumerate(split_parameters):
                        if cnt == 0:
                            pose_codes = split_parameters[cnt] #[4, 384]
                        else:
                            tmp = split_parameters[cnt] #[4,1024], [4,1024]
                            print(tmp.shape)
                            sigma_all.append(tmp)
                    print("sigma length", len(sigma_all)) #[[4,],[],[],[]]

                    for f_num in range(pose_codes.shape[0]):
                        shape_codes_inputs = shape_code.expand(points_flat.shape[0], -1)

                        pose_code = pose_codes[f_num].unsqueeze(0)  # 1,384
                        pose_codes_inputs = pose_code.expand(points_flat.shape[0], -1)

                        shape_pose_codes_inputs = torch.cat([shape_codes_inputs, pose_codes_inputs], 1)

                        pose_inputs = torch.cat([shape_pose_codes_inputs, points_flat], 1)

                        Sigma = torch.zeros(len(sigma_all), sigma_all[1].shape[1])
                        print("Sigma shape",Sigma.shape)   # 9,1024
                        for layer, sigma in enumerate(sigma_all):
                            print('sigma',sigma.shape)
                            Sigma[layer,:sigma.shape[1]] = sigma[f_num]
                        # print(Sigma)
                        sigma_t = Sigma.unsqueeze(0)
                        p_ref_warped = ft_decoder(pose_inputs, sigma_t, 1)

                        p_ref_warped = p_ref_warped.detach().cpu().numpy()

                        flow_mesh = t_mesh.copy()
                        flow_mesh.vertices = p_ref_warped

                        print("export flow mesh", f_num)
                        flow_mesh.export(save_path + f"/{f_num}.obj")

                        sys.stdout.flush()

            else:
                base_dir = f'./generation/diff/pose_pe-{all_frame}'
                os.makedirs(base_dir, exist_ok=True)
                torch.save(frames_gen, os.path.join(base_dir, 'frames_gen.pth'))

                for idx in range(number_of_samples_to_generate):
                    save_path = os.path.join(base_dir, str(idx))
                    os.makedirs(save_path, exist_ok=True)
                    shape_code = cond_in[idx] # 1,384
                    pose_codes = frames_gen[idx] #[4,384] -> [16,384]
                    # print("generated codes", pose_codes.shape, pose_codes)

                    t_mesh = deepsdf_utils.create_mesh_from_code(
                    self.shape_decoder, shape_code.unsqueeze(0), shape_codes_dim=shape_codes_dim, N=256, max_batch=int(2 ** 18))

                    t_mesh.export(save_path + "/t.obj")

                    p_ref = t_mesh.vertices.astype(np.float32)
                    points = torch.from_numpy(p_ref)[None, :].cuda()
                    points_flat = points.reshape(-1, 3)

                    for f_num in range(pose_codes.shape[0]):
                        shape_codes_inputs = shape_code.expand(points_flat.shape[0], -1)

                        pose_code = pose_codes[f_num].unsqueeze(0)  # 1,384
                        pose_codes_inputs = pose_code.expand(points_flat.shape[0], -1)

                        shape_pose_codes_inputs = torch.cat([shape_codes_inputs, pose_codes_inputs], 1)

                        pose_inputs = torch.cat([shape_pose_codes_inputs, points_flat], 1)

                        p_ref_warped, _ = self.pose_decoder(pose_inputs)
                        p_ref_warped = p_ref_warped.detach().cpu().numpy()

                        flow_mesh = t_mesh.copy()
                        flow_mesh.vertices = p_ref_warped

                        print("export flow mesh", f_num)
                        flow_mesh.export(save_path + f"/{f_num}.obj")
                        sys.stdout.flush()

