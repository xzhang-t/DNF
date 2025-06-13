#####################################################################################################################
# Base

exp_dir = "./experiments"
exp_version = "animal_diff"
train_dataset_name = 'data_split_diff'

stage = 'Diff'
mode = 'test'
only_shape = True
with_ft = True
gen_mesh = False
 
shape_dir = "SHAPE_MLP"
checkpoint = "latest"
ckpt = "latest"
ft_shape_dir = "SHAPE_FT_cp384_rk256"
if not only_shape:
    n_frame = 16
    diff_frame = 6
    pose_dir = "POSE_MLP"
    ft_pose_dir = "POSE_FT_cp768_rk512"

shape_ckpt_path = './experiments/animal_diff/SHAPE_Diff_wFT_bs48/checkpoints/last.ckpt'

shape_codes_dim = 384
pose_codes_dim = 384

shape_mlp_dim = [512, 512, 512, 77, 512, 512, 512, 512, 1]
pose_mlp_dim = [1024, 1024, 1024, 205, 1024, 1024, 1024, 1024, 3]

if with_ft:
    compress = True
    sigma_length = []
    if only_shape:
        k = 384
        rank = 256

        for dim in shape_mlp_dim:
            sigma_length.append(min(dim, k) + rank)
    else:
        k = 768
        rank = 512
        for dim in pose_mlp_dim:
            sigma_length.append(min(dim, k) + rank)

transformer_config = {
    "n_embd": 1280,
    "n_layer": 32,
    "n_head": 16,
    "split_policy": 'layer_by_layer',
    "use_global_residual": False,
    "condition": 'no' if only_shape else 'shape_latent',
    "cond_dim": 0 if only_shape else shape_codes_dim,
    "diff_frame":0 if only_shape else diff_frame,
}

t_embd = False

batch_size = 2
num_data_loader_threads = 16
epochs = 2000
timesteps = 500

diff_config = {
    "model_mean_type": 'START_X',
    "model_var_type": 'FIXED_LARGE',
    "loss_type": 'MSE',
    "add_inter": False,
    "inter_num": 0,
}

disable_wandb = True

lr = 0.0001
scheduler = True
scheduler_step = 50
normalization_factor = 1
accumulate_grad_batches = 1

continue_from = False
model_resume_path = None

shape_network_specs = {
    "dims": [512] * 8,
    "dropout": None,
    "dropout_prob": 0,
    "norm_layers": [0, 1, 2, 3, 4, 5, 6, 7],
    "latent_in": [4],
    "xyz_in_all": False,
    "latent_dropout": False,
    "weight_norm": True,
    "positional_enc": True,
    "n_positional_freqs": 8,
}

pose_network_specs = {
    "dims" : [1024] * 8,
    "dropout" : None,
    "dropout_prob" : 0,
    "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
    "latent_in" : [4],
    "xyz_in_all" : False,
    "latent_dropout" : False,
    "weight_norm" : True,
    "positional_enc": True,
    "n_positional_freqs": 8,
    "n_alpha_epochs": 0,
}
