import numpy as np
import os
import random
import torch
import torch.utils.data

def load_checkpoint(exp_name, checkpoint):
    exp_dir = '/cluster/falas/xzhang/experiments/animal'
    checkpoint_dir = os.path.join(exp_dir, exp_name, "checkpoints")

    if isinstance(checkpoint, int):
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{checkpoint}.tar")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint}_checkpoint.tar")

    if not os.path.exists(checkpoint_path):
        raise Exception(f'Other checkpoint {checkpoint_path} does not exist!')

    return torch.load(checkpoint_path, map_location=torch.device('cpu'))

class LatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        shape_dir,
        checkpoint,
        train_split,
        only_shape=True,
        pose_dir=None,
        n_frame=8,
        diff_frame=4,
    ):
        self.only_shape = only_shape
        checkpoint_shape = load_checkpoint(shape_dir, checkpoint)
        pretrained_shape_codes = checkpoint_shape['shape_codes']
        self.shape_codes = pretrained_shape_codes.cpu().detach().clone()
        print("Loaded SHAPE codes")
        print('Loaded checkpoint from:      {}'.format(shape_dir))
        self.train_split = train_split

        if not self.only_shape:
            checkpoint_pose = load_checkpoint(pose_dir, checkpoint)
            pretrained_pose_codes = checkpoint_pose['pose_codes']
            self.pose_codes = pretrained_pose_codes.cpu().detach().clone()
            print("Loaded POSE codes")
            print('Loaded checkpoint from:      {}'.format(pose_dir))
            self.n_frame = n_frame
            self.diff_frame = diff_frame
            self.n_seq = self.n_frame - self.diff_frame + 1

    def __len__(self):
        if self.only_shape:
            return len(self.train_split)
        else:
            return len(self.train_split) * self.n_seq

    def __getitem__(self, idx):
        if self.only_shape:
            idx_s = self.train_split[idx]
            shape_vec = self.shape_codes[idx_s].squeeze(0)
            return shape_vec, idx
        else:
            idx_s = idx // self.n_seq  #[0,1,2,3,4]
            shape_idx = self.train_split[idx_s]
            pose_idx_0 = shape_idx * self.n_frame + idx % self.n_seq
            assert shape_idx * self.n_frame <= pose_idx_0 < (shape_idx + 1) * self.n_frame, "not in one sequence!!"
            # print('seq idx, shape idx, pose idx', idx, shape_idx, pose_idx_0)
            shape_vec = self.shape_codes[shape_idx]

            pose_vec_list = []
            for i in range(0, self.diff_frame):
                # print('current pose idx', pose_idx_0+i)
                pose_vec = self.pose_codes[pose_idx_0+i].squeeze(0)
                pose_vec_list.append(pose_vec)
            pose_vec_list = torch.vstack(pose_vec_list)
            # print("pose vecs", pose_vec_list.shape) # 4,384
            return shape_vec, pose_vec_list, idx


class FTDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            shape_dir,
            checkpoint,
            ft_dir,
            train_split,
            sigma_length,
            only_shape=True,
            pose_dir=None,
            add_dir=None,
            n_frame=8,
            diff_frame=4,
            reverse=False
    ):
        self.only_shape = only_shape
        checkpoint_shape = load_checkpoint(shape_dir, checkpoint)
        pretrained_shape_codes = checkpoint_shape['shape_codes']
        self.shape_codes = pretrained_shape_codes.cpu().detach().clone()
        print("Loaded SHAPE codes")
        print('Loaded checkpoint from:      {}'.format(shape_dir))

        checkpoint_ft = load_checkpoint(ft_dir, checkpoint)
        self.sigma = checkpoint_ft['sigma'].cpu().detach().clone()
        self.sigma_length = sigma_length
        print("Loaded SIGMA")
        print('Loaded checkpoint from:      {}'.format(ft_dir))

        self.train_split = train_split

        if not self.only_shape:
            checkpoint_pose = load_checkpoint(pose_dir, checkpoint)
            pretrained_pose_codes = checkpoint_pose['pose_codes']
            self.pose_codes = pretrained_pose_codes.cpu().detach().clone()
            pose_code_dim = self.pose_codes.shape[-1]
            print("Loaded POSE codes")
            print('Loaded checkpoint from:      {}'.format(pose_dir))
            self.n_frame = n_frame
            self.diff_frame = diff_frame
            self.n_seq = self.n_frame - self.diff_frame + 1
            self.reverse = reverse
            if add_dir is not None:
                print("adding to 16 frames!!!!")
                pose_codes_f = self.pose_codes.reshape(-1, 8, pose_code_dim)

                checkpoint_add = load_checkpoint(add_dir, checkpoint)
                pose_codes_add = checkpoint_add['pose_codes'].cpu().detach().clone()
                pose_codes_add = pose_codes_add.reshape(-1, 8, pose_code_dim)
                pose_codes_all = torch.cat((pose_codes_f, pose_codes_add), dim=1)
                pose_codes_all = pose_codes_all.reshape(-1, pose_code_dim)
                self.pose_codes = pose_codes_all
                print("pose codes all", self.pose_codes.shape)

                # print("sigma", self.sigma.shape)
                layer_num = self.sigma.shape[-2]
                sigma_dim = self.sigma.shape[-1]
                sigma_f = self.sigma.reshape(-1, 8, layer_num, sigma_dim)
                sigma_add = checkpoint_add['sigma'].cpu().detach().clone()
                sigma_add = sigma_add.reshape(-1, 8, layer_num, sigma_dim)
                sigma_all = torch.cat((sigma_f, sigma_add), dim=1)
                sigma_all = sigma_all.reshape(-1, layer_num, sigma_dim)
                self.sigma = sigma_all
                print("sigma all", self.sigma.shape)

    def __len__(self):
        if self.only_shape:
            return len(self.train_split)
        else:
            if self.reverse:
                return len(self.train_split) * self.n_seq * 2
            else:
                return len(self.train_split) * self.n_seq

    def __getitem__(self, idx):
        # print("get item: ",torch.tensor(idx))
        weights = []
        if self.only_shape:
            # print('shape idx', idx)
            idx_s = self.train_split[idx]
            shape_vec = self.shape_codes[idx_s].squeeze(0)
            # print(shape_vec.shape)
            weights.append(shape_vec)
            sigma = self.sigma[idx_s]
            # print('sigma',sigma.shape)
            for i, sig in enumerate(sigma):
                sig_len = self.sigma_length[i]
                weights.append(sig[:sig_len])
                # print('sig',i, sig_len)
            weights = torch.hstack(weights)
            return weights, idx
        else:
            if self.reverse:
                # print("augment the dataset with reversing!!!")
                idx_f = idx // 2
                idx_s = idx_f // self.n_seq
                shape_idx = self.train_split[idx_s]
                pose_idx_0 = shape_idx * self.n_frame + idx % self.n_seq
                pose_idx_t = pose_idx_0 + self.diff_frame
                assert shape_idx * self.n_frame <= pose_idx_0
                assert pose_idx_t <= (shape_idx+1) * self.n_frame
                # print('seq idx, shape idx, pose idx', idx_f, shape_idx, pose_idx_0)

                shape_vec = self.shape_codes[shape_idx]

                if idx % 2 == 0:
                    # print("forward sequence")
                    for i in range(pose_idx_0, pose_idx_t):
                        frame_weights = []
                        pose_vec = self.pose_codes[i].squeeze(0)
                        frame_weights.append(pose_vec)
                        sigma = self.sigma[i]
                        for ith, sig in enumerate(sigma):
                            sig_len = self.sigma_length[ith]
                            frame_weights.append(sig[:sig_len])
                        frame_weights = torch.hstack(frame_weights)
                        weights.append(frame_weights)
                    weights = torch.vstack(weights)
                else:
                    # print("reverse sequence")
                    for i in range(pose_idx_t-1, pose_idx_0-1, -1):
                        frame_weights = []
                        pose_vec = self.pose_codes[i].squeeze(0)
                        frame_weights.append(pose_vec)
                        sigma = self.sigma[i]
                        for ith, sig in enumerate(sigma):
                            sig_len = self.sigma_length[ith]
                            frame_weights.append(sig[:sig_len])
                        frame_weights = torch.hstack(frame_weights)
                        weights.append(frame_weights)
                    weights = torch.vstack(weights)

            else:
                idx_s = idx // self.n_seq  # [0,1,2,3,4]
                shape_idx = self.train_split[idx_s]
                pose_idx_0 = shape_idx * self.n_frame + idx % self.n_seq
                assert shape_idx * self.n_frame <= pose_idx_0 <= (shape_idx+1) * self.n_frame - self.diff_frame, "not in one sequence!!"
                # print('seq idx, shape idx, pose idx', idx, shape_idx, pose_idx_0)
                shape_vec = self.shape_codes[shape_idx]

                for i in range(0, self.diff_frame):
                    frame_weights = []
                    # print('current pose idx', pose_idx_0 + i)
                    pose_vec = self.pose_codes[pose_idx_0 + i].squeeze(0)
                    frame_weights.append(pose_vec)
                    sigma = self.sigma[pose_idx_0 + i]
                    for ith, sig in enumerate(sigma):
                        sig_len = self.sigma_length[ith]
                        frame_weights.append(sig[:sig_len])
                        # print('sig', i, ith, sig_len)
                    frame_weights = torch.hstack(frame_weights)
                    weights.append(frame_weights)

                weights = torch.vstack(weights)

            return shape_vec, weights, idx










