#####################################################################################################################
# Base


exp_dir = "/cluster/falas/xzhang/experiments"
exp_version = "animal_diff"

stage = 'Diff'
mode = 'train'
only_shape = False
with_ft = False
gen_mesh = False
train_dataset_name = 'data_split_diff'
# train_dataset_name = 'data_split_diff_one'

shape_dir = "2024-10-07__NPMs__SHAPE_nss0.7_uni0.3__bs16__lr-0.0005-0.0005-0.001-0.001-0.0001-0.0001_intvl500__s384-512-8l__p384-1024-8l__woSE3__wShapePosEnc__wPosePosEnc__woDroutS__woDroutP__wWNormS__wWNormP__ON__4d-animal-new"
# shape_dir = "2024-10-27__NPMs__SHAPE__MLP__bs16__L1__wDroutS0.1__ON__DT4D_con+"
checkpoint = "latest"
ckpt = "latest"
ft_shape_dir = "2024-10-21__NPMs__SHAPE__bs16__SHAPE_FT_cp384_rk256__L1__ON__DT4D"
# ft_dir = "2024-10-28__NPMs__SHAPE__MLP__bs8__SHAPE_FT_cp384_rk256_SVD__L1__wDroutS0.1__ON__DT4D_con+"

load_cond = False
calc_metrics = False
if mode == 'test':
    if not only_shape:
        # best_model_save_path = './experiments/animal_diff/2024-10-22__NPMs__POSE__Diff__woFT__bs32/checkpoints/last.ckpt'
        best_model_save_path = '/cluster/falas/xzhang/experiments/animal_diff/2024-10-27__NPMs__POSE__Diff__woFT__bs48/checkpoints/last-epoch=1999-train_loss=0.01.ckpt'
        if with_ft:
            # best_model_save_path = './experiments/animal_diff/2024-10-22__NPMs__POSE__Diff__bs32/checkpoints/last.ckpt'
            # best_model_save_path = '/cluster/falas/xzhang/experiments/animal_diff/2024-10-27__NPMs__POSE__Diff__wFT__bs48/checkpoints/last-epoch=1999-train_loss=0.00.ckpt'
            best_model_save_path = '/cluster/falas/xzhang/experiments/animal_diff/2024-11-09__NPMs__POSE__Diff__wFT__bs1__ON__data_split_diff_one/checkpoints/last-epoch=99-train_loss=0.00.ckpt'

        load_cond = False
        gen_path = './generation/diff/shape_ft/sample_x_0s.pth'
        shape_layers = [384, 640, 640, 640, 333, 640, 640, 640, 640, 257]

    if only_shape:
        best_model_save_path = '/cluster/falas/xzhang/experiments/animal_diff/2024-10-23__NPMs__SHAPE__Diff__woFT__bs48/checkpoints/last-epoch=1999-train_loss=0.02.ckpt'
        if with_ft:
            best_model_save_path = '/cluster/falas/xzhang/experiments/animal_diff/2024-10-23__NPMs__SHAPE__Diff__wFT__bs48/checkpoints/last-epoch=1968-train_loss=0.00.ckpt'

cond_jitter = False
reverse = False
if not only_shape:
    n_frame = 8
    diff_frame = 4
    reverse = False
    cond_jitter = False
    pose_dir = "2024-10-16__NPMs__POSE__bs8__MSE__ON__DT4D_con-motion"
    ft_pose_dir = "2024-10-18__NPMs__POSE__bs12__POSE_FT_cp768_rk512__L1__ON__DT4D_con-motion"
    model_resume_path = None
    # model_resume_path = '/cluster/falas/xzhang/experiments/animal_diff/2024-11-03__NPMs__POSE__Diff__wFT__bs48__ON__data_split_diff2__with_T/checkpoints/last-epoch=824-train_loss=0.00.ckpt'
    add_dir = None
    # add_dir = '2024-11-09__NPMs__POSE__MLP__bs16__POSE_FT_cp768_rk512_SVD__L1__woDroutS__ON__DT4D_con-motion__fit_s'

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


batch_size = 72
num_data_loader_threads = 32
epochs = 1500
timesteps = 500

diff_config = {
    "model_mean_type": 'START_X',
    "model_var_type": 'FIXED_LARGE',
    "loss_type": 'MSE',
}

disable_wandb = True

lr = 0.0001
scheduler = True
scheduler_step = 50
normalization_factor = 1
# learning_rate_schedules = {
#  "diff": {
#         "type": "step",
#         "initial": 0.0001,
#         "interval": 1000,
#         "factor": 0.5,
#     },
# }
accumulate_grad_batches = 1

continue_from = False

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
