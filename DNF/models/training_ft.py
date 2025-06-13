from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import shutil
import random
from collections import OrderedDict

from models.shape_decoder import ShapeDecoder
from models.pose_decoder import PoseDecoder
import utils.deepsdf_utils as deepsdf_utils 
import utils.nnutils as nnutils
from models.dictionary_pose import Dict_pose_acc
from models.dictionary import Dict_acc

import configs_train.config_train_DT4D as cfg

# For DataParallel: https://discuss.pytorch.org/t/how-could-i-train-on-multi-gpu-and-infer-with-single-gpu/22838/6

class Trainer():

    def __init__(
        self, 
        debug,
        device, 
        train_dataset,
        exp_dir, exp_name,
        train_to_augmented = None
    ):
        self.sigma_regularization = cfg.do_sigma_regularization
        self.do_code_regularization = cfg.do_code_regularization
        self.do_svd_regularization = cfg.do_svd_regularization
        self.sigma_reg_lambda = cfg.sigma_reg_lambda
        self.svd_reg_lambda = cfg.svd_reg_lambda
        self.only_shape = cfg.only_shape
        # self.with_normal = cfg.with_normal

        ###############################################################################################
        # Model
        ###############################################################################################

        # Code dim
        self.shape_codes_dim = cfg.shape_codes_dim
        self.pose_codes_dim = cfg.pose_codes_dim

        self.num_identities = train_dataset.get_num_identities()
        self.num_train_samples = len(train_dataset)

        # CODES
        self.shape_codes = torch.ones(self.num_identities, 1, self.shape_codes_dim).normal_(0, 1.0 / self.shape_codes_dim).to(device)
        self.shape_codes.requires_grad = False

        self.train_dataset = train_dataset

        self.num_points_sdf = self.train_dataset.get_num_samples_per_shape()['sdf']
        print("Num samples sdf:", self.num_points_sdf)

        self.shape_decoder = ShapeDecoder(self.shape_codes_dim, **cfg.shape_network_specs).to(device)
        for p in self.shape_decoder.parameters():
            p.requires_grad = False

        self.exp_dir = exp_dir
        print("exp dir",exp_dir)
        self.exp_name = exp_name

        self.checkpoint = None

        self.exp_path = os.path.join(exp_dir, exp_name)
        self.checkpoints_dir = os.path.join(self.exp_path, 'checkpoints')
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        # Copy config.py over
        config_path = os.path.join("configs_train", f"config_train_{cfg.config_dataset}.py")
        shutil.copy(config_path, os.path.join(self.exp_path, f"config_{cfg.config_dataset}.py"))

        self.writer = SummaryWriter(os.path.join(self.exp_path, 'summary'))

        self.vis_dir = os.path.join(self.exp_path, 'vis')
        self.device = device
        ###############################################################################################
        ###############################################################################################
        self.bias = []
        self.weights = []
        self.U = []
        self.Sigma_ori = []
        self.Vt = []
        self.rank = cfg.rank
        self.half = cfg.half
        self.k = cfg.k

        self.min_vec = torch.ones(self.num_points_sdf * cfg.batch_size, 1).cuda() * (-cfg.clamping_distance)
        self.max_vec = torch.ones(self.num_points_sdf * cfg.batch_size, 1).cuda() * cfg.clamping_distance

        self.load_from_other()
        print("Decomposing shape decoder with SVD")
        self.MLP_SVD()

        ## shape of Sigma [n, layer_num, len(U)+rank]
        self.Sigma = torch.zeros(len(self.U), self.U[1].shape[1]+self.rank).to(device)
        nn.init.normal_(self.Sigma, mean=0.0, std=0.001)
        for i, sigma in enumerate(self.Sigma_ori):
            print(i, sigma.shape)
            self.Sigma[i,:sigma.shape[0]] = sigma
            print(i, self.Sigma[i].shape)
        self.Sigma = self.Sigma.unsqueeze(0).repeat(self.shape_codes.shape[0], 1, 1)
        # print(self.Sigma.shape)
        self.Sigma.requires_grad = True

        self.ft_decoder = Dict_acc(self.rank, self.half, positional_enc=cfg.shape_network_specs['positional_enc']).to(device)
        self.ft_decoder.init_para(self.bias, self.U, self.Vt)
        self.ft_decoder.to(device)

        self.gpu_num = torch.cuda.device_count()
        if self.gpu_num > 1:
            print()
            print(f"Using {self.gpu_num} GPUs")
            print()
            self.ft_decoder = torch.nn.DataParallel(self.ft_decoder)
            self.ft_decoder.to(device)

        for name, para in self.ft_decoder.named_parameters():
            if 'res' not in name:
                para.requires_grad_(False)
            else:
                print(name)
        pg = [p for p in self.ft_decoder.parameters() if p.requires_grad]

        #####################################################################################
        # Set up optimizer.
        #####################################################################################
        lr_schedule_ft_decoder = nnutils.StepLearningRateSchedule(cfg.learning_rate_schedules['ft_decoder'])
        lr_schedule_sigma = nnutils.StepLearningRateSchedule(cfg.learning_rate_schedules['sigma'])

        self.lr_schedules = [
            lr_schedule_ft_decoder,
            lr_schedule_sigma,
        ]
        learnable_params = [
            {
                "params": pg,
                "lr": lr_schedule_ft_decoder.get_learning_rate(0),
            },
            {
                "params": self.Sigma,
                "lr": lr_schedule_sigma.get_learning_rate(0),
            },
        ]

        if cfg.optimizer == 'Adam':
            self.optimizer = optim.Adam(learnable_params)
        else:
            raise Exception("Others optimizers are not available yet.")
        print()

        self.eval_every = cfg.eval_every
        self.interval = cfg.interval

        #####################################################################################
        # Create dirs
        #####################################################################################
        self.lambdas = cfg.lambdas_sdf
        self.eps = 0
        self.lamb = 0
        assert self.lambdas['ref'] > 0 or self.lambdas['flow'] > 0, "What do you expect to train?"

        self.criterion_l1 = deepsdf_utils.SoftL1()
        self.tanh = nn.Tanh()


    def train_model(self):
        start = 0

        for epoch in range(start, cfg.epochs + cfg.epochs_extra):
            ############################################################
            sum_loss_total = 0
            sum_loss_ref = 0
            sum_loss_svd = 0
            # print(self.Sigma.shape)

            self.current_epoch = epoch
            ############################################################################

            print()
            print(f'Epoch {epoch} - {self.exp_name}')

            train_data_loader = self.train_dataset.get_loader(shuffle=True)

            nnutils.adjust_learning_rate(self.lr_schedules, self.optimizer, epoch)

            ############################################################
            # Store checkpoint
            ############################################################
            if epoch % self.eval_every == 0:
                # Store latest checkpoint
                self.save_special_checkpoint(epoch, "latest")

                # Store a checkpoint every 10 epochs
                self.save_checkpoint(epoch)
            ############################################################
            # TRAIN
            ############################################################
            num_samples_in_batch = len(train_data_loader)

            loop = tqdm(train_data_loader)
            for batch in loop:
                loss_dict = self.train_step(batch, epoch)

                loss_total = loss_dict['total']
                loss_ref = loss_dict['loss_ref']
                loss_svd = loss_dict['loss_svd']
                loop.set_postfix(loss=loss_total)

                sum_loss_total += loss_total
                sum_loss_ref += loss_ref
                sum_loss_svd += loss_svd

            sum_loss_total = sum_loss_total / num_samples_in_batch
            sum_loss_ref = sum_loss_ref / num_samples_in_batch
            sum_loss_svd = sum_loss_svd / num_samples_in_batch

            # if epoch % self.eval_every == 0:
            self.writer.add_scalar('train/0_loss', sum_loss_total, epoch)
            self.writer.add_scalar('train/1_ref', sum_loss_ref, epoch)
            self.writer.add_scalar('train/2_svd', sum_loss_svd, epoch)

            print(
                "Current loss: {:.4f} - "
                "ref: {:.4f} ({:.4f}) - "
                "svd: {:.4f}".format(
                    sum_loss_total,
                    sum_loss_ref, self.lambdas['ref'] * sum_loss_ref,
                    sum_loss_svd
                )
            )

            ######### Visualize ###########
            # if epoch % self.interval == 0:
            #     # idx = np.random.randint(0, self.num_identities)
            #     # print(idx)
            #     idx = 424
            #     os.makedirs(self.vis_dir, exist_ok=True)
            #     if not os.path.isfile(self.vis_dir + f'/{idx}_t_ori.obj'):
            #         ref_mesh = deepsdf_utils.create_mesh(
            #             self.shape_decoder, self.shape_codes, identity_ids=[idx], shape_codes_dim=self.shape_codes_dim,
            #             N=256, max_batch=int(2 ** 18))
            #         # save_path = os.path.join(self.vis_dir, str(epoch) + '-' + str(idx))
            #         # os.makedirs(save_path, exist_ok=True)
            #         # ref_mesh.export(save_path + '/t_ori.obj')
            #         ref_mesh.export(self.vis_dir + f'/{idx}_t_ori.obj')
            #
            #     sig = self.Sigma[idx,...].unsqueeze(0)
            #
            #     ref_mesh = deepsdf_utils.create_mesh(
            #         self.ft_decoder, self.shape_codes, identity_ids=[idx], sigma=sig, bs=1, shape_codes_dim=self.shape_codes_dim,
            #         N=256, max_batch=int(2 ** 18))
            #     # save_path = os.path.join(self.vis_dir, str(epoch) + '-' + str(idx))
            #     # os.makedirs(save_path, exist_ok=True)
            #     ref_mesh.export(self.vis_dir + f'/{idx}_t_ft_{epoch}.obj')

    def load_from_other(self):
        checkpoint = self.load_checkpoint(cfg.init_from, cfg.checkpoint)
        print("Loaded SHAPE decoder")
        self.shape_decoder.load_state_dict(checkpoint['model_state_dict_shape_decoder'])
        print("Loaded SHAPE codes")
        self.shape_codes = checkpoint['shape_codes'].to(self.device).detach().clone()
        print()
        print('Loaded checkpoint from:      {}'.format(cfg.init_from))

    def load_checkpoint(self, exp_name, checkpoint):
        checkpoint_dir = os.path.join(self.exp_dir, exp_name, "checkpoints")

        if isinstance(checkpoint, int):
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{checkpoint}.tar")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint}_checkpoint.tar")

        if not os.path.exists(checkpoint_path):
            raise Exception(f'Other checkpoint {checkpoint_path} does not exist!')

        return torch.load(checkpoint_path)

    def MLP_SVD(self):
        torch.nn.utils.remove_weight_norm(self.shape_decoder.lin0)
        torch.nn.utils.remove_weight_norm(self.shape_decoder.lin1)
        torch.nn.utils.remove_weight_norm(self.shape_decoder.lin2)
        torch.nn.utils.remove_weight_norm(self.shape_decoder.lin3)
        torch.nn.utils.remove_weight_norm(self.shape_decoder.lin4)
        torch.nn.utils.remove_weight_norm(self.shape_decoder.lin5)
        torch.nn.utils.remove_weight_norm(self.shape_decoder.lin6)
        torch.nn.utils.remove_weight_norm(self.shape_decoder.lin7)

        state_dict = self.shape_decoder.state_dict()
        layers = []
        layer_names = []

        for i, l in enumerate(state_dict):
            layer_names.append(l)
            if 'bias' in l:
                self.bias.append(state_dict[l])
            else:
                self.weights.append(state_dict[l])

        for i in range(len(self.weights)):
            w = self.weights[i]
            layers.append(w)
            u, sigma, vt = torch.linalg.svd(layers[i])
            self.U.append(u)
            self.Sigma_ori.append(torch.log(sigma))  # Sigma = ln(sigma) [fs,259]
            self.Vt.append(vt)

        for i in range(len(self.U)):
            if cfg.compress:
                k = min(self.k, self.Sigma_ori[i].shape[0])
                self.U[i] = self.U[i][:, :k]
                self.Sigma_ori[i] = self.Sigma_ori[i][:k]
                self.Vt[i] = self.Vt[i][:k, :]
                print("shape of U Sigma Vt, ", i, self.U[i].shape, self.Sigma_ori[i].shape, self.Vt[i].shape)
            else:
                if self.U[i].shape[1] > self.Vt[i].shape[0]:
                    self.U[i] = self.U[i][:, :self.Vt[i].shape[0]]
                elif self.U[i].shape[1] < self.Vt[i].shape[0]:
                    self.Vt[i] = self.Vt[i][:self.U[i].shape[0]]
                print("shape of U Sigma Vt, ", i, self.U[i].shape, self.Sigma_ori[i].shape, self.Vt[i].shape)


    def train_step(self, batch, epoch):
        self.ft_decoder.train()

        for param in self.ft_decoder.parameters():
            param.grad = None

        self.Sigma.grad = None

        loss, loss_dict = self.compute_loss(batch, epoch)
        loss.backward()
        self.optimizer.step()

        # Project latent vectors onto sphere
        if cfg.code_bound is not None:
            deepsdf_utils.project_latent_codes_onto_sphere(self.shape_codes, cfg.code_bound)
            if not self.only_shape:
                deepsdf_utils.project_latent_codes_onto_sphere(self.pose_codes, cfg.code_bound)

        return loss_dict

    def compute_loss(self, batch, epoch):
        device = self.device

        ################################################################
        # Get data
        ################################################################

        ref = batch.get('ref')
        indices = batch.get('idx')
        shape_sigma_batch = self.Sigma[indices, ...]
        # print("sigma shape",shape_sigma_batch.shape)  # bs, layer, dim

        # motion_names = batch.get('motion_name')

        #############################################################################
        # SDF points
        #############################################################################
        p_sdf_ref = ref['points_sdf'][0].to(device)  # [bs, N, 3]

        batch_size = p_sdf_ref.shape[0]

        sdf_data = p_sdf_ref.reshape(self.num_points_sdf * batch_size, 4)
        p_sdf = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        #############################################################################
        # Initialize losses
        #############################################################################
        loss = torch.tensor((0.0)).to(device)

        loss_ref_item = 0.0
        loss_svd_item = 0.0

        ##########################################################################################
        ##########################################################################################
        # Forward pass
        ##########################################################################################
        ##########################################################################################

        # Get shape codes for batch samples
        assert torch.all(indices < self.shape_codes.shape[0]), f"{indices} vs {self.shape_codes.shape[0]}"
        shape_codes_batch = self.shape_codes[indices, ...]  # [bs, 1, C]

        ##########################################################################################
        # A) Reconstruct shape in reference frame
        ##########################################################################################
        if self.lambdas['ref'] > 0:
            # Extent latent code to all sampled points
            shape_codes_repeat = shape_codes_batch.expand(-1, self.num_points_sdf, -1)  # [bs, N, C]
            shape_codes_inputs = shape_codes_repeat.reshape(-1, self.shape_codes_dim)  # [bs*N, C]

            shape_inputs = torch.cat([shape_codes_inputs, p_sdf], 1)

            # Truncate groundtruth sdf
            if cfg.enforce_minmax:
                sdf_gt = deepsdf_utils.threshold_min_max(sdf_gt, self.min_vec, self.max_vec)

            pred_sdf = self.ft_decoder(shape_inputs, shape_sigma_batch, int(batch_size/self.gpu_num))  # [bs*N, 1]
            # print(pred_sdf.shape)

            # Truncate predicted sdf
            if cfg.enforce_minmax:
                pred_sdf = deepsdf_utils.threshold_min_max(pred_sdf, self.min_vec, self.max_vec)

            pred_sdf = pred_sdf / self.max_vec
            sdf_gt = sdf_gt / self.max_vec
            # print(pred_sdf)
            # print(sdf_gt)
            loss_ref, l1_ref_raw = self.criterion_l1(pred_sdf, sdf_gt, self.eps)
            loss_ref = loss_ref * (1.0 + self.lamb * torch.sign(sdf_gt) * torch.sign(sdf_gt - pred_sdf))
            loss_ref = torch.mean(loss_ref)

            loss += self.lambdas['ref'] * loss_ref

            loss_ref_item = l1_ref_raw.item()
            # print('ref loss',loss_ref)

        if self.do_svd_regularization:
            svd_loss = 0.0
            # svd_loss_all = 0.0
            loss_l1 = torch.nn.L1Loss(reduction="sum")
            for layer in range(len(self.U) - self.half):
                if isinstance(self.ft_decoder, torch.nn.DataParallel):
                    UU = self.ft_decoder.module.U[layer].weight
                    VV = self.ft_decoder.module.Vt[layer].weight
                    uu = self.ft_decoder.module.res_u[layer].weight
                    vv = self.ft_decoder.module.res_vt[layer].weight
                else:
                    UU = self.ft_decoder.U[layer].weight
                    VV = self.ft_decoder.Vt[layer].weight
                    uu = self.ft_decoder.res_u[layer].weight
                    vv = self.ft_decoder.res_vt[layer].weight

                if uu.shape[0] < uu.shape[1]:
                    uu = uu.t()
                if vv.shape[0] > vv.shape[1]:
                    vv = vv.t()
                svd_u = loss_l1(torch.mm(uu.t(), uu), torch.eye(uu.shape[1]).cuda())
                svd_v = loss_l1(torch.mm(vv, vv.t()), torch.eye(vv.shape[0]).cuda())
                # print('svd u', svd_u)
                # print('svd v', svd_v)
                svd_loss += svd_u + svd_v

            # svd_loss = svd_loss / len(self.U) * self.svd_reg_lambda * min(1, epoch / 100)
            svd_loss = svd_loss / len(self.U) * self.svd_reg_lambda
            # print('svd loss', svd_loss)
            loss += svd_loss
            loss_svd_item = svd_loss.item()

        # Prepare dict of losses
        loss_dict = {
            'total': loss.item(),
            'loss_ref': loss_ref_item,
            'loss_svd': loss_svd_item,
        }
        return loss, loss_dict

    def save_base(self, path, epoch):
        if isinstance(self.ft_decoder, torch.nn.DataParallel):
            ft_decoder_state_dict = self.ft_decoder.module.state_dict()
        else:
            ft_decoder_state_dict = self.ft_decoder.state_dict()
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict_ft_decoder': ft_decoder_state_dict,
                'sigma': self.Sigma.detach().cpu(),
                'optimizer_state_dict': self.optimizer.state_dict()
            },
            path
        )

    def save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}.tar')
        if not os.path.exists(path):
            self.save_base(path, epoch)

    def save_special_checkpoint(self, epoch, special_name):
        path = os.path.join(self.checkpoints_dir, f'{special_name}_checkpoint.tar')
        self.save_base(path, epoch)


    def load_latest_checkpoint(self):

        checkpoints = [m for m in os.listdir(self.checkpoints_dir)]

        if len(checkpoints) == 0:
            print()
            print('No checkpoints found at {}'.format(self.checkpoints_dir))
            return 0

        # If we're here, we have at least 1 checkpoint
        latest_checkpoint_path = os.path.join(self.checkpoints_dir, "latest_checkpoint.tar")

        if not os.path.exists(latest_checkpoint_path):
            raise Exception(f'Latest checkpoint {latest_checkpoint_path} does not exist!')

        checkpoint = torch.load(latest_checkpoint_path)
        self.checkpoint = checkpoint

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.shape_decoder.load_state_dict(checkpoint['model_state_dict_shape_decoder'])
        self.pose_decoder.load_state_dict(checkpoint['model_state_dict_pose_decoder'])

        assert checkpoint['shape_codes'].shape == self.shape_codes.shape
        self.shape_codes = checkpoint['shape_codes'].to(self.device).detach().clone()
        self.shape_codes.requires_grad = False if cfg.freeze_shape_codes else True

        assert checkpoint['pose_codes'].shape == self.pose_codes.shape
        self.pose_codes = checkpoint['pose_codes'].to(self.device).detach().clone()
        self.pose_codes.requires_grad = False if cfg.freeze_pose_codes else True

        epoch = checkpoint['epoch']

        print("Loaded epoch", epoch)
        print("Optim shape code", self.shape_codes.requires_grad)
        print("Optim pose code", self.pose_codes.requires_grad)
        
        print()
        print('Loaded checkpoint from: {}'.format(latest_checkpoint_path))

        print()
        print("Looking good???")
        print()
        
        return epoch

    def load_checkpoint(self, exp_name, checkpoint):
        checkpoint_dir = os.path.join(self.exp_dir, exp_name, "checkpoints")

        if isinstance(checkpoint, int): 
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{checkpoint}.tar")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint}_checkpoint.tar")

        if not os.path.exists(checkpoint_path):
            raise Exception(f'Other checkpoint {checkpoint_path} does not exist!')

        return torch.load(checkpoint_path)


    def get_new_state_dict(state_dict):
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v

        return new_state_dict
    
