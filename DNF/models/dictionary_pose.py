import torch.nn as nn
import torch
import torch.nn.functional as F
import  numpy as np
import time
from utils import embedder
from torch.autograd import grad

class Dict_pose(nn.Module):
    def __init__(
        self,
        bias,
        U,
        Sigma,
        Vt,
        latent_in = [4],
        dropout_prob = 0.0,
        dropout = [0, 1, 2, 3, 4, 5, 6, 7],
        positional_enc = False,
        n_positional_freqs = 8,
        n_alpha_epochs = 0
    ):
        super(Dict_pose, self).__init__()
        self.U_w = U
        self.U = nn.ModuleList([])
        self.Sigma = nn.ParameterList([])
        self.Vt_w = Vt
        self.Vt = nn.ModuleList([])
        self.latent_in = latent_in
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.num_layers = len(U)
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

        for i in range(len(U)):
            self.U.append(
                nn.Linear(U[i].shape[1], U[i].shape[0], bias=True)
            )
            self.Sigma.append(
                nn.Parameter(Sigma[i])
            )
            self.Vt.append(
                nn.Linear(Vt[i].shape[1], Vt[i].shape[0], bias=False)
            )

    #     线性层参数初始化
    #     print(len(self.U),len(self.Sigma),len(self.Vt))

        for i in range(len(self.U)):
            self.U[i].weight = nn.Parameter(U[i])
            self.U[i].bias = nn.Parameter(bias[i])
            self.Vt[i].weight = nn.Parameter(Vt[i])

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                n_positional_freqs, input_dims=3
            )
            self.n_alpha_epochs = n_alpha_epochs
            self.alpha_const = n_positional_freqs / n_alpha_epochs if n_alpha_epochs > 0 else self.n_positional_freqs

    def forward(self, input):
        if hasattr(self, "pos_embedder"):
            xyz = input[:, -3:]
            input_pos_embed = self.pos_embedder(xyz, self.n_positional_freqs)
            x = torch.cat([input[:, :-3], input_pos_embed], 1)
            input_embed = x.clone()
        else:
            x = input
        # print("self.num_layers", self.num_layers)

        for i, Sig in enumerate(self.Sigma):

            if i in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                    # print(x.shape)
                else:
                    x = torch.cat([x, input], 1)

            # print(x)
            # print("!!!",i, x.shape)
            x = self.Vt[i](x)
            # print(self.Vt[i].weight,self.Vt[i].bias)
            # print("after vt",i, x.shape,x)
            # x = torch.mm(x,Sig)
            # print(x.shape,Sig.shape)
            Sig = Sig.unsqueeze(0)  # 1,259
            sig_repeat = Sig.expand(x.shape[0], -1)  # 30000,259
            x = x * sig_repeat
            # print("after s",i, x.shape,x)
            x = self.U[i](x)
            # print("after u and bias ",i, x.shape,x)

            # print(i, x.shape, x)
            if i < self.num_layers - 1:
                x = self.relu(x)
                # print("after relu",i,x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
                    # print("after dropout : ", i, x.shape, x)
            # print("final output",i, x.shape,x)

        # x = self.th(x)
        xyz_warped = input[:, -3:] + x
        return xyz_warped


class Dict_exp_pose(nn.Module):
    def __init__(
        self,
        bias,
        U,
        Sigma,
        Vt,
        latent_in=[4],
        dropout_prob=0.05,
        dropout= None,
        positional_enc=False,
        n_positional_freqs=8,
        n_alpha_epochs=0,
    ):
        super(Dict_exp_pose, self).__init__()
        self.U_w = U
        self.U = nn.ModuleList([])
        self.Sigma = nn.ParameterList([]) ### self.sigma = ln(sigma)
        self.Vt_w = Vt
        self.Vt = nn.ModuleList([])
        self.latent_in = latent_in
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.num_layers = len(U)
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

        for i in range(len(U)):
            self.U.append(
                nn.Linear(U[i].shape[1], U[i].shape[0], bias=True)
            )
            self.Sigma.append(
                nn.Parameter(Sigma[i])
                # nn.Parameter(torch.log(Sigma[i]))
            )
            self.Vt.append(
                nn.Linear(Vt[i].shape[1], Vt[i].shape[0], bias=False)
            )

    #     initialize linear layer
    #     print(len(self.U),len(self.Sigma),len(self.Vt))

        for i in range(len(self.U)):
            self.U[i].weight = nn.Parameter(U[i])
            self.U[i].bias = nn.Parameter(bias[i])
            self.Vt[i].weight = nn.Parameter(Vt[i])

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                n_positional_freqs, input_dims=3
            )
            self.n_alpha_epochs = n_alpha_epochs
            self.alpha_const = n_positional_freqs / n_alpha_epochs if n_alpha_epochs > 0 else self.n_positional_freqs

    def forward(self, input):
        Sigma_m = []
        for i, sigma in enumerate(self.Sigma):
            # Sigma_m.append(self.get_diag(sigma, self.U_w[i].shape[1], self.Vt_w[i].shape[0]))
            Sigma_m.append(torch.exp(sigma))
            # print(i,sigma_m.shape)

        if hasattr(self, "pos_embedder"):
            xyz = input[:, -3:]
            input_pos_embed = self.pos_embedder(xyz, self.alpha_const)
            x = torch.cat([input[:, :-3], input_pos_embed], 1)
            input_embed = x.clone()
        else:
            x = input
        # print("self.num_layers", self.num_layers)
        # print("initial x",x)

        for i, Sig in enumerate(Sigma_m):
            if i in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                else:
                    x = torch.cat([x, input], 1)

            # print("initial x",x)
            x = self.Vt[i](x)
            # print(x.shape)
            # print("after vt", i, x.shape, x)
            # print(self.Vt[i].weight,self.Vt[i].bias)
            # print("if Sig<=0: ", False in Sig.ge(0))
            # x = torch.mm(x,Sig)   # x [30000,259]
            Sig = Sig.unsqueeze(0)  # 1,259
            sig_repeat = Sig.expand(x.shape[0], -1)  # 30000,259
            x = x * sig_repeat
            # print(x.shape)
            # print("after s",i, x.shape,x)
            x = self.U[i](x)
            # print(x.shape)
            # print("after u and bias ",i, x.shape,x)

            # print(i, x.shape, x)
            if i < self.num_layers - 1:
                x = self.relu(x)
                # print("after relu",i,x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
                    # print("after dropout : ", i, x.shape, x)
            # print("final output",i, x.shape, x)

        # x = self.th(x)
        # xyz_warped = input[:, -3:] + x
        xyz_warped = x
        return xyz_warped


class Dict_exp_pose_multi(nn.Module):
    def __init__(
        self,
        batch_size,
        bias,
        U,
        Sigma,
        Vt,
        latent_in = [4],
        dropout_prob = 0.05,
        dropout = [0, 1, 2, 3, 4, 5, 6, 7],
        positional_enc=False,
        n_positional_freqs=8,
        n_alpha_epochs=0,
    ):
        super(Dict_exp_pose_multi, self).__init__()
        self.bs = batch_size
        self.U_w = U
        self.U = nn.ModuleList([])
        self.Sigma = nn.ParameterList([]) ### self.sigma = ln(sigma)
        self.Vt_w = Vt
        self.Vt = nn.ModuleList([])
        self.latent_in = latent_in
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.num_layers = len(U)
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

        input_dim = 3
        output_dim = 3

        for i in range(len(U)):
            self.U.append(
                nn.Linear(U[i].shape[1], U[i].shape[0], bias=True)
            )
            self.Sigma.append(
                nn.Parameter(Sigma[i])
                # nn.Parameter(Sigma[i].unsqueeze(0).repeat(self.bs, 1))
                # nn.Parameter(torch.log(Sigma[i]))
            )
            # print("shape of simga ",i)
            # print(Sigma[i].unsqueeze(0).repeat(self.bs, 1).shape)
            self.Vt.append(
                nn.Linear(Vt[i].shape[1], Vt[i].shape[0], bias=False)
            )

    #     线性层参数初始化
    #     print(len(self.U),len(self.Sigma),len(self.Vt))

        for i in range(len(self.U)):
            self.U[i].weight = nn.Parameter(U[i])
            self.U[i].bias = nn.Parameter(bias[i])
            self.Vt[i].weight = nn.Parameter(Vt[i])

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                n_positional_freqs, input_dims=input_dim
            )
            input_dim = pos_embedder_out_dim
            self.n_alpha_epochs = n_alpha_epochs
            self.alpha_const = n_positional_freqs / n_alpha_epochs if n_alpha_epochs > 0 else self.n_positional_freqs



    def forward(self, input):
        Sigma_m = []
        for i, sigma in enumerate(self.Sigma):
            Sigma_m.append(torch.exp(sigma))
        #     # print(sigma.shape)
        #     # Sigma_m.append(self.get_diag(sigma, self.U_w[i].shape[1], self.Vt_w[i].shape[0]))
        #     Sigma_m.append(self.get_diag(sigma))
        #     # print(i,sigma_m.shape)


        if hasattr(self, "pos_embedder"):
            xyz = input[:, :, -3:].reshape(-1,3)
            bs = input.shape[0]
            # print("xyz",xyz.shape)
            # print(self.alpha_const)
            input_pos_embed = self.pos_embedder(xyz, self.alpha_const)
            # print(input_pos_embed)
            dim = input_pos_embed.shape[1]
            input_pos_embed = input_pos_embed.reshape(bs,-1,dim)
            # print("input_pos_embed",input_pos_embed.shape)
            # print("!!!!!",input[:, :, :-3].shape,input_pos_embed.shape)
            x = torch.cat([input[:, :, :-3], input_pos_embed], 2)
            # print("pos embed", input_pos_embed)
            # print(x.shape)
            input_embed = x.clone()

            # xyz = input[0, :, -3:]
            # print("xyz",xyz.shape)
            # input_pos_embed = self.pos_embedder(xyz, self.n_positional_freqs)
            # x = torch.cat([input[0, :, :-3], input_pos_embed], 1)
            # input_embed = x.clone()

        else:
            x = input

        # print("self.num_layers", self.num_layers)
        # print(x.shape) bs,n,c

        for i, Sig in enumerate(Sigma_m):
            if i in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 2)
                else:
                    x = torch.cat([x, input], 2)

            # print("initial x", i, x.shape,x)
            x = self.Vt[i](x)

            # print("after vt ", i, x.shape,x)
            # print(self.Vt[i].weight,self.Vt[i].bias)
            # print("if Sig<=0: ", False in Sig.ge(0))
            # print("forward, ",x.shape,Sig.shape)

            # x = torch.bmm(x,Sig)
            Sig = Sig.unsqueeze(1) #bs,1,259
            sig_repeat = Sig.expand(-1, x.shape[1], -1) #bs,30000,259
            x = x * sig_repeat

            # print("after s",i, x.shape,x)
            x = self.U[i](x)

            # print("after u and bias ",i, x.shape,x)

            # print(i, x.shape)
            if i < self.num_layers - 1:
                x = self.relu(x)
                # print("after relu",i,x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
                    # print("after dropout : ", i, x.shape, x)
            # print("final output",i, x.shape,x)

        # xyz_warped = input[:, :, -3:] + x
        # x = self.th(x)
        # print(xyz_warped)
        # print("final", x)
        xyz_warped = x
        return xyz_warped


class AdaptiveDict(nn.Module):
    def __init__(
        self,
        bias,
        U,
        Sigma,
        Vt,
        rank,
        latent_in = [4],
        dropout_prob = 0.2,
        dropout = [0, 1, 2, 3, 4, 5, 6, 7],
        norm_layers = [0, 1, 2, 3, 4, 5, 6, 7],

    ):
        super(AdaptiveDict, self).__init__()
        self.U_w = U
        self.U = nn.ModuleList([])
        self.Sigma = nn.ParameterList([])
        self.Vt_w = Vt
        self.Vt = nn.ModuleList([])
        self.rank = rank
        # print("rank!!",self.rank)
        self.res_u = nn.ModuleList([])
        self.res_sigma = nn.ParameterList([])
        self.res_vt = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.latent_in = latent_in
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.norm_layers = norm_layers
        self.num_layers = len(U)
        self.relu = nn.ReLU()
        self.th = nn.Tanh()


        # print("ini_U",torch.isnan(U).any())
        # print("ini_V", torch.isnan(Vt).any())
        for i in range(len(U)):
            self.U.append(
                nn.Linear(U[i].shape[1],U[i].shape[0],bias=True)
            )
            self.Sigma.append(
                nn.Parameter(Sigma[i])
            )
            self.Vt.append(
                nn.Linear(Vt[i].shape[1], Vt[i].shape[0],bias=False)
            )
            rk = min(self.rank,Sigma[i].shape[0])
            # print("rk",rk)
            self.res_u.append(
                # nn.utils.weight_norm(nn.Linear(rk,U[i].shape[0],bias=False))
                nn.Linear(rk, U[i].shape[0], bias=False)
            )
            self.res_sigma.append(
                nn.Parameter(torch.zeros(rk))
            )
            self.res_vt.append(
                # nn.utils.weight_norm(nn.Linear(Vt[i].shape[1], rk, bias=False))
                nn.Linear(Vt[i].shape[1], rk, bias=False)
            )
            # self.bn.append(
            #     nn.LayerNorm(U[i].shape[0])
            # )

    #     print(len(self.U),len(self.Sigma),len(self.Vt))

        for i in range(len(self.U)):
            self.U[i].weight = nn.Parameter(U[i])
            self.U[i].bias = nn.Parameter(bias[i])
            self.Vt[i].weight = nn.Parameter(Vt[i])
            # nn.init.kaiming_normal_(self.res_u[i].weight.data)
            # nn.init.kaiming_normal_(self.res_vt[i].weight.data)
            nn.init.normal_(self.res_u[i].weight,std=np.sqrt(1/512))
            nn.init.normal_(self.res_vt[i].weight, std=np.sqrt(1/512))
            # nn.utils.weight_norm(self.res_u[i])
            # nn.utils.weight_norm(self.res_vt[i])

    def get_diag(self, sigma, width, length):
        sigma_m = torch.diag_embed(sigma)
        if length < width:
            zeros = torch.zeros(length, width - length).cuda()
            sigma_m = torch.cat((sigma_m, zeros), axis=1)
        elif length > width:
            zeros = torch.zeros(length - width, width).cuda()
            sigma_m = torch.cat((sigma_m, zeros), axis=0)
        return sigma_m

    def forward(self, input):
        Sigma_m = []
        for i, sigma in enumerate(self.Sigma):
            Sigma_m.append(self.get_diag(sigma, self.U_w[i].shape[1], self.Vt_w[i].shape[0]))
        res_Sigma_m = []
        for i, sigma in enumerate(self.res_sigma):
            res_Sigma_m.append(self.get_diag(sigma,self.rank,self.rank))

        x = input

        for i, Sig in enumerate(Sigma_m):
            if i in self.latent_in:
                x = torch.cat([x, input], 1)

            x1 = self.Vt[i](x)
            x1 = torch.mm(x1,Sig)
            x1 = self.U[i](x1)

            # x2 = self.res[i](x)
            x2 = self.res_vt[i](x)
            # print("x2_1!!!!", torch.isnan(x2).any())
            x2 = torch.mm(x2,res_Sigma_m[i])
            # print(res_Sigma_m[i])
            # print("x2_2!!!!", torch.isnan(x2).any())
            # print("max",torch.max(x2))
            x2 = self.res_u[i](x2)
            # print("x2_3!!!!",torch.isnan(x2).any())
            # if (
            #     self.norm_layers is not None
            #     and i in self.norm_layers
            # ):
            #     x2 = self.bn[i](x2)

            x = x1 + x2

            # x =   self.U[i](torch.mm(self.Vt[i](x),Sig)) + self.res[i](input)

            if i < self.num_layers - 1:
                x = self.relu(x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
                # print("relu and dropout!!!!", torch.isnan(x).any())

        x = self.th(x)
        # print("x final!!!!", torch.isnan(x).any())

        return x


class AdaptiveDict_exp(nn.Module):
    def __init__(
        self,
        bias,
        U,
        Sigma,
        Vt,
        rank,
        latent_in = [4],
        dropout_prob = 0.05,
        # dropout = [0, 1, 2, 3, 4, 5, 6, 7],
        dropout = None,
        norm_layers = [0, 1, 2, 3, 4, 5, 6, 7],
        positional_enc=False,
        n_positional_freqs=8,

    ):
        super(AdaptiveDict_exp, self).__init__()
        self.U_w = U
        self.U = nn.ModuleList([])
        self.Sigma = nn.ParameterList([])
        self.Vt_w = Vt
        self.Vt = nn.ModuleList([])
        self.rank = rank
        # print("rank!!",self.rank)
        self.res_u = nn.ModuleList([])
        self.res_sigma = nn.ParameterList([])
        self.res_vt = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.latent_in = latent_in
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.norm_layers = norm_layers
        self.num_layers = len(U)
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

        # print("ini_U",torch.isnan(U).any())
        # print("ini_V", torch.isnan(Vt).any())
        for i in range(len(U)):
            self.U.append(
                nn.Linear(U[i].shape[1],U[i].shape[0],bias=True)
            )
            self.Sigma.append(
                nn.Parameter(Sigma[i])
            )
            self.Vt.append(
                nn.Linear(Vt[i].shape[1], Vt[i].shape[0],bias=False)
            )
            rk = min(self.rank,Sigma[i].shape[0])
            # print("rk",rk)
            self.res_u.append(
                # nn.utils.weight_norm(nn.Linear(rk,U[i].shape[0],bias=False))
                nn.Linear(rk, U[i].shape[0], bias=False)
            )
            self.res_sigma.append(
                # nn.Parameter(torch.full((rk,),-10.0))
                nn.Parameter(torch.zeros(rk))
            )
            self.res_vt.append(
                # nn.utils.weight_norm(nn.Linear(Vt[i].shape[1], rk, bias=False))
                nn.Linear(Vt[i].shape[1], rk, bias=False)
            )
            # self.bn.append(
            #     nn.LayerNorm(U[i].shape[0])
            # )

    #     线性层参数初始化
    #     print(len(self.U),len(self.Sigma),len(self.Vt))

        for i in range(len(self.U)):
            self.U[i].weight = nn.Parameter(U[i])
            self.U[i].bias = nn.Parameter(bias[i])
            self.Vt[i].weight = nn.Parameter(Vt[i])

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                self.n_positional_freqs, input_dims=3, i=0
            )

    def get_diag(self, sigma, width, length):
        sigma_m = torch.diag_embed(torch.exp(sigma))
        if length < width:
            zeros = torch.zeros(length, width - length).cuda()
            sigma_m = torch.cat((sigma_m, zeros), axis=1)
        elif length > width:
            zeros = torch.zeros(length - width, width).cuda()
            sigma_m = torch.cat((sigma_m, zeros), axis=0)
        return sigma_m

    def forward(self, input):

        Sigma_m = []
        for i, sigma in enumerate(self.Sigma):
            Sigma_m.append(self.get_diag(sigma, self.U_w[i].shape[1], self.Vt_w[i].shape[0]))
        res_Sigma_m = []
        for i, sigma in enumerate(self.res_sigma):
            res_Sigma_m.append(self.get_diag(sigma,self.rank,self.rank))

        if hasattr(self, "pos_embedder"):
            xyz = input[:, :, -3:]
            input_pos_embed = self.pos_embedder(xyz, self.n_positional_freqs)
            x = torch.cat([input[:, :-3], input_pos_embed], 1)
            input_embed = x.clone()
        else:
            x = input

        for i, Sig in enumerate(Sigma_m):
            if i in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                else:
                    x = torch.cat([x, input], 1)

            x1 = self.Vt[i](x)
            # print("if Sig<=0: ", False in Sig.ge(0))
            x1 = torch.mm(x1,Sig)
            x1 = self.U[i](x1)

            # x2 = self.res[i](x)
            x2 = self.res_vt[i](x)
            # print("x2_1!!!!", torch.isnan(x2).any())
            # print("if res_Sig<=0: ", False in res_Sigma_m[i].ge(0))
            x2 = torch.mm(x2,res_Sigma_m[i])
            # print(res_Sigma_m[i])
            # print("x2_2!!!!", torch.isnan(x2).any())
            # print("max",torch.max(x2))
            x2 = self.res_u[i](x2)
            # print("x2_3!!!!",torch.isnan(x2).any())
            # if (
            #     self.norm_layers is not None
            #     and i in self.norm_layers
            # ):
            #     x2 = self.bn[i](x2)

            x = x1 + x2

            # x =   self.U[i](torch.mm(self.Vt[i](x),Sig)) + self.res[i](input)

            if i < self.num_layers - 1:
                x = self.relu(x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
                # print("relu and dropout!!!!", torch.isnan(x).any())

        x = self.th(x)
        # print("x final!!!!", torch.isnan(x).any())

        return x


class AdaptiveDict_exp_pose_multi(nn.Module):
    def __init__(
        self,
        batch_size,
        bias,
        U,
        Sigma,
        Vt,
        rank,
        half,
        latent_in = [4],
        dropout_prob = 0.05,
        dropout = [0, 1, 2, 3, 4, 5, 6, 7],
        norm_layers = [0, 1, 2, 3, 4, 5, 6, 7],
        positional_enc=False,
        n_positional_freqs=8,
        n_alpha_epochs=0,
    ):
        super(AdaptiveDict_exp_pose_multi, self).__init__()
        self.bs = batch_size
        self.U_w = U
        self.U = nn.ModuleList([])
        self.Sigma = nn.ParameterList([])
        self.Vt_w = Vt
        self.Vt = nn.ModuleList([])
        self.rank = rank
        self.half = half
        # print("rank!!",self.rank)
        self.res_u = nn.ModuleList([])
        self.res_sigma = nn.ParameterList([])
        self.res_vt = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.latent_in = latent_in
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.norm_layers = norm_layers
        self.num_layers = len(U)
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

        # print("ini_U",torch.isnan(U).any())
        # print("ini_V", torch.isnan(Vt).any())
        for i in range(len(U)):
            self.U.append(
                nn.Linear(U[i].shape[1],U[i].shape[0],bias=True)
            )
            self.Sigma.append(
                nn.Parameter(Sigma[i])
            )
            self.Vt.append(
                nn.Linear(Vt[i].shape[1], Vt[i].shape[0],bias=False)
            )

            # print("rk",rk)
            if i >= self.half:
                rk = min(self.rank, Sigma[i].shape[0])
                self.res_u.append(
                    # nn.utils.weight_norm(nn.Linear(rk,U[i].shape[0],bias=False))
                    nn.Linear(rk, U[i].shape[0], bias=False)
                )
                self.res_sigma.append(
                    nn.Parameter(torch.full((self.bs, rk),-3.0))
                    # nn.Parameter(torch.zeros((self.bs,rk)))
                )
                self.res_vt.append(
                    # nn.utils.weight_norm(nn.Linear(Vt[i].shape[1], rk, bias=False))
                    nn.Linear(Vt[i].shape[1], rk, bias=False)
                )
            # self.bn.append(
            #     nn.LayerNorm(U[i].shape[0])
            # )

    #     线性层参数初始化
    #     print(len(self.U),len(self.Sigma),len(self.Vt))

        for i in range(len(self.U)):
            self.U[i].weight = nn.Parameter(U[i])
            self.U[i].bias = nn.Parameter(bias[i])
            self.Vt[i].weight = nn.Parameter(Vt[i])

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                n_positional_freqs, input_dims=3
            )
            self.n_alpha_epochs = n_alpha_epochs
            self.alpha_const = n_positional_freqs / n_alpha_epochs if n_alpha_epochs > 0 else self.n_positional_freqs

    def forward(self, input):
        Sigma_m = []
        for i, sigma in enumerate(self.Sigma):
            Sigma_m.append(torch.exp(sigma))
            # Sigma_m.append(self.get_diag(sigma))

        res_Sigma_m = []
        for i, sigma in enumerate(self.res_sigma):
            res_Sigma_m.append(torch.exp(sigma))
            # res_Sigma_m.append(self.get_diag(sigma))

        if hasattr(self, "pos_embedder"):
            xyz = input[:, :, -3:].reshape(-1, 3)
            bs = input.shape[0]
            input_pos_embed = self.pos_embedder(xyz, self.alpha_const)
            dim = input_pos_embed.shape[1]
            input_pos_embed = input_pos_embed.reshape(bs, -1, dim)
            x = torch.cat([input[:, :, :-3], input_pos_embed], 2)
            input_embed = x.clone()
        else:
            x = input

        for i, Sig in enumerate(Sigma_m):
            if i in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 2)
                else:
                    x = torch.cat([x, input], 2)

            x1 = self.Vt[i](x)  #259, 259    259,259+()  bs*30000*259   -> bs*30000*259 bmm bs*259*259
            # print("if Sig<=0: ", False in Sig.ge(0))
            # x1 = torch.bmm(x1, Sig)  #bs, 259+()     bs*30000*259
            Sig = Sig.unsqueeze(1)  # bs,1,259
            sig_repeat = Sig.expand(-1, x.shape[1], -1)  # bs,30000,259
            x1 = x1 * sig_repeat

            x1 = self.U[i](x1)   # 259+(),512

            # x2 = self.res[i](x)
            if i >= self.half:
                x2 = self.res_vt[i - self.half](x)  # 259*2
                # print("if res_Sig<=0: ", False in res_Sigma_m[i].ge(0))
                # x2 = torch.bmm(x2, res_Sigma_m[i])  # bs,2
                res_sig = res_Sigma_m[i - self.half].unsqueeze(1)  # bs,1,2
                res_sig_repeat = res_sig.expand(-1, x.shape[1], -1)  # bs,30000,2
                x2 = x2 * res_sig_repeat
                # print(res_Sigma_m[i])
                # print("max",torch.max(x2))
                x2 = self.res_u[i - self.half](x2)  # 2,512
                # print("x2_3!!!!",torch.isnan(x2).any())
            # if (
            #     self.norm_layers is not None
            #     and i in self.norm_layers
            # ):
            #     x2 = self.bn[i](x2)
                x = x1 + x2
            else:
                x = x1

            # x =   self.U[i](torch.mm(self.Vt[i](x),Sig)) + self.res[i](input)

            if i < self.num_layers - 1:
                x = self.relu(x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
                # print("relu and dropout!!!!", torch.isnan(x).any())

        # x = self.th(x)
        # print("x final!!!!", torch.isnan(x).any())
        # xyz_warped = input[:, :, -3:] + x
        xyz_warped = x

        return xyz_warped


class Dict_pose_acc(nn.Module):
    def __init__(
        self,
        rank,
        half,
        latent_in = [4],
        dropout_prob = 0.0,
        dropout = [0, 1, 2, 3, 4, 5, 6, 7],
        norm_layers = [0, 1, 2, 3, 4, 5, 6, 7],
        positional_enc=False,
        n_positional_freqs=8,
        n_alpha_epochs=0,
    ):
        super(Dict_pose_acc, self).__init__()
        self.U = nn.ModuleList([])
        self.Vt = nn.ModuleList([])
        self.rank = rank
        self.half = half
        # print("rank!!",self.rank)
        self.res_u = nn.ModuleList([])
        # self.res_sigma = nn.ParameterList([])
        self.res_vt = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.latent_in = latent_in
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.norm_layers = norm_layers
        self.num_layers = 0
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                n_positional_freqs, input_dims=3
            )
            self.n_alpha_epochs = n_alpha_epochs
            self.alpha_const = n_positional_freqs / n_alpha_epochs if n_alpha_epochs > 0 else self.n_positional_freqs

    def init_para(self, bias, U, Vt):
        self.num_layers = len(U)

        for i in range(len(U)):
            self.U.append(
                nn.Linear(U[i].shape[1], U[i].shape[0], bias=True)
            )
            self.Vt.append(
                nn.Linear(Vt[i].shape[1], Vt[i].shape[0], bias=False)
            )

            self.U[i].weight = nn.Parameter(U[i])
            self.U[i].bias = nn.Parameter(bias[i])
            self.Vt[i].weight = nn.Parameter(Vt[i])

            # print("rk",rk)
            if i >= self.half and self.rank:
                rk = self.rank
                self.res_u.append(
                    nn.Linear(rk, U[i].shape[0], bias=False)
                )
                # nn.init.orthogonal_(self.res_u[i].weight)
                nn.init.normal_(self.res_u[i].weight, mean=0.0, std=0.001)
                self.res_vt.append(
                   nn.Linear(Vt[i].shape[1], rk, bias=False)
                )
                # nn.init.orthogonal_(self.res_vt[i].weight)
                nn.init.normal_(self.res_vt[i].weight, mean=0.0, std=0.001)

    def forward(self, input, Sigma, bs):
        xyz = input[:, -3:]

        if hasattr(self, "pos_embedder"):
            input_pos_embed = self.pos_embedder(xyz, self.alpha_const)
            x = torch.cat([input[:, :-3], input_pos_embed], 1)
            input_embed = x.clone()
        else:
            x = input

        for i in range(len(self.U)):
            if i in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                else:
                    x = torch.cat([x, input], 1)

            ori_len = self.Vt[i].out_features

            x1 = self.Vt[i](x)
            Sig = torch.exp(Sigma[:, i, :ori_len])
            Sig = Sig.unsqueeze(1)
            sig_repeat = Sig.expand(-1, int(x1.shape[0] / bs), -1)
            sig_repeat = sig_repeat.reshape(-1, sig_repeat.shape[-1])
            x1 = x1 * sig_repeat

            x1 = self.U[i](x1)   # 259+(),512

            if i >= self.half and self.rank:
                x2 = self.res_vt[i - self.half](x)
                res_sig = torch.exp(Sigma[:, i, ori_len:ori_len+self.rank])
                res_sig = res_sig.unsqueeze(1)
                res_sig_repeat = res_sig.expand(-1, int(x1.shape[0] / bs), -1)  # bs,30000,2
                res_sig_repeat = res_sig_repeat.reshape(-1, res_sig_repeat.shape[-1])
                x2 = x2 * res_sig_repeat
                x2 = self.res_u[i - self.half](x2)
                x = x1 + x2
            else:
                x = x1

            if i < self.num_layers - 1:
                x = self.relu(x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        xyz_warped = xyz + x

        return xyz_warped

