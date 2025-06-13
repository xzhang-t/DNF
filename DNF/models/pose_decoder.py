#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# Updates by Pablo Palafox 2021

from functools import wraps
from torch import nn, einsum
import torch
import torch.nn.functional as F
import kornia
import numpy as np
from torch.autograd import grad

from utils import embedder
from utils import geometry_utils

class PoseDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        latent_dropout=False,
        positional_enc=False,
        n_positional_freqs=8,
        n_alpha_epochs=80,
    ):
        super(PoseDecoder, self).__init__()

        input_dim = 3
        output_dim = 3

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                n_positional_freqs, input_dims=input_dim
            )
            input_dim = pos_embedder_out_dim
            self.n_alpha_epochs = n_alpha_epochs
            self.alpha_const = n_positional_freqs / n_alpha_epochs if n_alpha_epochs > 0 else self.n_positional_freqs

        dims = [latent_size + input_dim] + dims + [output_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm


        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and l in self.norm_layers:
                # print(l, in_dim, dims[l + 1])
                print(l, dims[l], out_dim)
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                # print(l, in_dim, dims[l+1])
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))

            if (not weight_norm) and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))

        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout

    # input: N x (L+3)
    def forward(self, input, epoch=None):
        xyz = input[:, -3:]  #(n,3)-> (n,f*3) -> (n,f,3)

        if hasattr(self, "pos_embedder"):
            alpha = self.alpha_const * epoch if self.n_alpha_epochs > 0 else self.alpha_const
            input_pos_embed = self.pos_embedder(xyz, alpha)
            x = torch.cat([input[:, :-3], input_pos_embed], 1)
            # print("pos embed", input_pos_embed)
            input_embed = x.clone()
        else:
            if input.shape[1] > 3 and self.latent_dropout:
                latent_vecs = input[:, :-3]
                latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
                x = torch.cat([latent_vecs, xyz], 1)
            else:
                x = input

        for l in range(0, self.num_layers - 1):
            # print("initial x", l, x.shape, x)
            
            lin = getattr(self, "lin" + str(l))
            
            if l in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                else:
                    x = torch.cat([x, input], 1)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)

            x = lin(x)
            # print("after lin ",l, x.shape,x)
            
            if l < self.num_layers - 2:
                if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm:
                    bn = getattr(self, "bn" + str(l))
                    x = bn(x)

                x = self.relu(x)
                # print("after relu", l, x)

                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
            # print("final output", l, x.shape, x)

        # Apply predicted translation
        xyz_warped = xyz + x

        return xyz_warped, x


class PoseDecoder_T(nn.Module):
    def __init__(
            self,
            latent_size,
            dims,
            dropout=None,
            dropout_prob=0.0,
            norm_layers=(),
            latent_in=(),
            weight_norm=False,
            xyz_in_all=None,
            latent_dropout=False,
            positional_enc=False,
            n_positional_freqs=8,
            n_alpha_epochs=80,
    ):
        super(PoseDecoder_T, self).__init__()

        input_dim = 4
        output_dim = 3

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                n_positional_freqs, input_dims=input_dim
            )
            input_dim = pos_embedder_out_dim
            self.n_alpha_epochs = n_alpha_epochs
            self.alpha_const = n_positional_freqs / n_alpha_epochs if n_alpha_epochs > 0 else self.n_positional_freqs

        dims = [latent_size + input_dim] + dims + [output_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3
            # if l in latent_in:
            #     in_dim = dims[l] + dims[0]
            # else:
            #     in_dim = dims[l]
            if weight_norm and l in self.norm_layers:
                # print(l, in_dim, dims[l + 1])
                print(l, dims[l], out_dim)
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
                # setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(in_dim, dims[l+1])))
            else:
                # print(l, in_dim, dims[l+1])
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))
                # setattr(self, "lin" + str(l), nn.Linear(in_dim, dims[l+1]))

            if (not weight_norm) and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))
                # setattr(self, "bn" + str(l), nn.LayerNorm(dims[l+1]))

        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout

    # input: N x (L+3)
    def forward(self, input, epoch=None):
        # xyz = input[:, -3:]  # (n,3)-> (n,f*3) -> (n,f,3)
        xyz = input[:, -3:]
        txyz = input[:, -4:]

        if hasattr(self, "pos_embedder"):
            alpha = self.alpha_const * epoch if self.n_alpha_epochs > 0 else self.alpha_const
            input_pos_embed = self.pos_embedder(txyz, alpha)
            x = torch.cat([input[:, :-4], input_pos_embed], 1)
            # print("pos embed", input_pos_embed)
            input_embed = x.clone()
        else:
            if input.shape[1] > 4 and self.latent_dropout:
                latent_vecs = input[:, :-4]
                latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
                x = torch.cat([latent_vecs, txyz], 1)
            else:
                x = input

        # print("!!!!!!!! input")
        # print(input_embed)

        for l in range(0, self.num_layers - 1):
            # print("initial x", l, x.shape, x)

            lin = getattr(self, "lin" + str(l))

            if l in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                else:
                    x = torch.cat([x, input], 1)

            x = lin(x)
            # print("after lin ",l, x.shape,x)

            if l < self.num_layers - 2:
                if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm:
                    bn = getattr(self, "bn" + str(l))
                    x = bn(x)

                x = self.relu(x)
                # print("after relu", l, x)

                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
            # print("final output", l, x.shape, x)

        # Apply predicted translation
        xyz_warped = xyz + x

        return xyz_warped, x

    def gradient(self, inputs, outputs):
        d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
        points_grad = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0][:, -3:]
        return points_grad

