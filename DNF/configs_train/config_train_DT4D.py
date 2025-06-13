#####################################################################################################################
# SET ME!!!
config_dataset = "DT4D"
ROOT = "/cluster/andram/xzhang/DT4D_con"
# SET ME!!!
####################################################################################################################
stage = 'MLP'
data_base_dir = f"{ROOT}"
exp_dir = "/cluster/falas/xzhang/experiments"

#####################################################################################################################
# MODELS
#####################################################################################################################

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

k_shape_network_specs = {
    "layer": 0,
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

k_pose_network_specs = {
    "layer": 0,
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

shape_codes_dim = 384
pose_codes_dim = 384

#####################################################################################################################
# DATA
#####################################################################################################################
mse = False

############################################################################################
############################################################################################
only_shape = False  # Set to True when bootstrapping the ShapeMLP
shape_ft = False
pose_ft = True
if only_shape:
    rank = 256
else:
    rank = 512
half = 0
############################################################################################
############################################################################################

# if only_shape:
#     cache_data = True
# else:
#     cache_data = False
cache_data = True

# if shape_ft:
#     cache_data = False

t_embd = False

compress = True
if only_shape:
    k = 384
else:
    k = 768
# --------------------------------------------------------------------------------------- #
# Choose a name for the folder where we will be storing experiments
exp_version = "animal"
# --------------------------------------------------------------------------------------- #

############################################################################################
############################################################################################
#############
### SHAPE ###
#############
shape_dataset_name = ROOT.split('/')[-1]

#############
### POSE ###
#############
pose_dataset_name = ROOT.split('/')[-1] + '-motion'

train_dataset_name = shape_dataset_name if only_shape else pose_dataset_name
############################################################################################
############################################################################################

#####################################################################################################################
#####################################################################################################################

batch_size = 16 if only_shape else 10 # the higher, the better, of course
n_frame = 16
if shape_ft:
    batch_size = 16
if pose_ft:
    batch_size = 16
    # frame_size = 4
num_workers = 4 if only_shape else 8
gpu_num = 1

## SDF samples
num_sdf_samples = 25000 if only_shape else 0

## Flow samples
sample_flow_dist = [0.7, 0.3]
sample_flow_sigmas = [0.0, 0.02]  # [0.01, 0.1] #[0.002, 0.01]
num_flow_points = 20000 if not only_shape else 0

sample_info = {
    'sdf': {
        'num_points': num_sdf_samples,
        'res_points': 30000
    },
    'sdf_all': {
        'num_points': num_sdf_samples,
    },
    'flow': {
        'dist': sample_flow_dist,
        'sigmas': sample_flow_sigmas,
        'num_points': num_flow_points,
    }
}

weight_correspondences = False
weights_dict = {
    'use_procrustes': False,
    'min_rigidity_distance': 0.1,
    'rigidity_scale': 2.0,
    'rigidity_max_weight': 2.0,
}

#####################################################################################################################
# TRAINING OPTIONS
#####################################################################################################################


# SDF OPTIONS
enforce_minmax = True
clamping_distance = 0.1

# Set it to None to disable this "sphere" normalization | 1.0
code_bound = None

epochs = 1001 if only_shape else 301
if pose_ft:
    epochs = 401
epochs_extra = 0 if only_shape else 0
eval_every = 200 if only_shape else 100
interval = 100 if only_shape else 50

optimizer = 'Adam'

##################################################################################
##################################################################################
# Set to True if you wanna start training from a given checkpoint, which
# you need to specify below
# If False, we check whether we "continue_from"
init_from = True
resume = False
continue_from = False
##################################################################################
##################################################################################

# If we're only training the shape latent space, we typically wanna start from scratch
if only_shape:
    init_from = False
    if shape_ft or pose_ft:
        init_from = True


# Set the init checkpoint if necessary
if init_from:
    # continue_from = False  # Since we will initialize from a checkpoint, we set continue_from to False

    # By default, init_from_pose is set to False, but you can set it to a certain checkpoint if you want to start the Pose MLP from a checkpoint different than the Shape MLP
    init_from_pose = False

    # This is where you specify your shape MLP checkpoint, to then train the Pose MLP
    # init_from = "<SET YOUR CHECKPOINT NAME HERE>"
    # checkpoint = 0 # And set the epoch of the checkpoint

    init_from = "2024-10-07__NPMs__SHAPE_nss0.7_uni0.3__bs16__lr-0.0005-0.0005-0.001-0.001-0.0001-0.0001_intvl500__s384-512-8l__p384-1024-8l__woSE3__wShapePosEnc__wPosePosEnc__woDroutS__woDroutP__wWNormS__wWNormP__ON__4d-animal-new"
    checkpoint = "latest"  # And set the epoch of the checkpoint

    if pose_ft:
        init_from_pose = "2024-11-25__NPMs__POSE__MLP__bs10__MSE__woDroutS__ON__DT4D_con-motion"
        checkpoint_pose = 'latest'

if resume:
    continue_from = ""


######## finetune resume
resume = False
resume_dir = ''

##################################################################################
# Load modules
if shape_ft:
    load_shape_decoder = True
    load_shape_codes = True
    freeze_shape_decoder = True
    freeze_shape_codes = True
    load_pose_decoder = False
    load_pose_codes = False
    freeze_pose_decoder = False
    freeze_pose_codes = False
    freeze_pose_encoder = False
elif pose_ft:
    load_shape_decoder = True
    load_shape_codes = True
    freeze_shape_decoder = True
    freeze_shape_codes = True
    load_pose_decoder = True
    load_pose_codes = True
    freeze_pose_decoder = True
    freeze_pose_codes = True
    freeze_pose_encoder = True
else:
    load_shape_decoder = True if not only_shape else False
    load_shape_codes = True if not only_shape else False
    freeze_shape_decoder = True if not only_shape else False
    freeze_shape_codes = True if not only_shape else False

    load_pose_decoder = False
    load_pose_codes = False
    freeze_pose_decoder = False
    freeze_pose_codes = False
    freeze_pose_encoder = False


if continue_from:
    assert not only_shape
    # In the current implementation, we only allow to "continue_from" if we're learning the pose space
    load_shape_decoder = True
    load_shape_codes = True
    freeze_shape_decoder = True
    freeze_shape_codes = True

    load_pose_decoder = True
    load_pose_codes = True
    freeze_pose_decoder = False
    freeze_pose_codes = False
    freeze_pose_encoder = False


lambdas_sdf = {
    'ref': 0 if not only_shape else 1,
    "flow": 1 if not only_shape else 0,
}

if lambdas_sdf['ref'] > 0:
    assert num_sdf_samples > 0

if lambdas_sdf['flow'] > 0:
    assert num_flow_points > 0

do_code_regularization = True
if shape_ft or pose_ft:
    do_code_regularization = False
    do_sigma_regularization = False
    do_svd_regularization = False
shape_reg_lambda = 1e-4
pose_reg_lambda = 1e-4
sigma_reg_lambda = 1e-5
svd_reg_lambda = 1e-6
# svd_reg_lambda = 0

lr_dict = {
    "shape_decoder": 0.0005,
    "pose_decoder": 0.0005,
    "shape_codes": 0.001,
    "pose_codes": 0.001,
    "ft_decoder": 0.0001,
    "sigma": 0.0001,
}

learning_rate_schedules = {
    "shape_decoder": {
        "type": "step",
        "initial": lr_dict['shape_decoder'],
        "interval": interval,
        "factor": 0.5,
    },
    "pose_decoder": {
        "type": "step",
        "initial": lr_dict['pose_decoder'],
        "interval": interval,
        "factor": 0.5,
    },
    "shape_codes": {
        "type": "step",
        "initial": lr_dict['shape_codes'],
        "interval": interval,
        "factor": 0.5,
    },
    "pose_codes": {
        "type": "step",
        "initial": lr_dict['pose_codes'],
        "interval": interval,
        "factor": 0.5,
    },
    "ft_decoder": {
        "type": "step",
        "initial": lr_dict['ft_decoder'],
        "interval": interval,
        "factor": 0.5,
    },
    "sigma": {
        "type": "step",
        "initial": lr_dict['sigma'],
        "interval": interval,
        "factor": 0.5,
    },

}
