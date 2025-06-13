#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# Updates by Pablo Palafox 2021

import torch.nn as nn
import torch
import torch.nn.functional as F

from utils import embedder


class ShapeDecoder(nn.Module):
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
        use_tanh=False,
        latent_dropout=False,
        positional_enc=False,
        n_positional_freqs=8,
    ):
        super(ShapeDecoder, self).__init__()

        input_dim = 3
        output_dim = 1
        ## latent code + xyz 256+3 256+51
        ## SDF (xyz+latent) -> sdf value
        ## (xyz_encoded + latent) -> sdf value

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                self.n_positional_freqs, input_dims=input_dim, i=0
            )
            dims = [latent_size + pos_embedder_out_dim] + dims + [output_dim]
        else:
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
                    out_dim -= input_dim

            if weight_norm and l in self.norm_layers:
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))

            if (not weight_norm) and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))

        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input, epoch=None):
        xyz = input[:, -3:]

        if hasattr(self, "pos_embedder"):
            input_pos_embed = self.pos_embedder(xyz, self.n_positional_freqs)
            x = torch.cat([input[:, :-3], input_pos_embed], 1)
            input_embed = x.clone()
        else:
            if input.shape[1] > 3 and self.latent_dropout:
                latent_vecs = input[:, :-3]
                latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
                x = torch.cat([latent_vecs, xyz], 1)
            else:
                x = input

        for l in range(0, self.num_layers - 1):
            
            lin = getattr(self, "lin" + str(l))
            
            if l in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                else:
                    x = torch.cat([x, input], 1)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)

            x = lin(x)
            # print(l, x.shape, x)

            if l < self.num_layers - 2:
                if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm:
                    bn = getattr(self, "bn" + str(l))
                    x = bn(x)

                x = self.relu(x)
                # print("after relu", l, x)

                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
                    # print("after dropout : ", l, x.shape, x)

        if hasattr(self, "th"):
            x = self.th(x)

        return x

