import os
import argparse
import torch
import datasets.sdf_dataset_4d_new as sdf_dataset
from models import training_t,training_ft_pose,training_ft
import json
import utils.utils as utils

# import config as cfg
import configs_train.config_train_DT4D as cfg
print(f"Loaded DT4D config.")

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-n', '--extra_name', default="", type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

###############################################################################################
# Dataset and Experiment dirs
###############################################################################################

data_dir = cfg.data_base_dir
exp_dir  = os.path.join(cfg.exp_dir, cfg.exp_version)

print()
print("DATA_DIR:", data_dir)
print("EXP_DIR:", exp_dir)
print()

splits_dir = './animals/data_split_motion.json'
with open(splits_dir, "r") as f:
    data_split = json.load(f)


print("TRAIN DATASET...")
train_dataset = sdf_dataset.SDFDataset(
    # data_dir=os.path.join(data_dir,'meshes'),
    data_dir=data_dir,
    # data_dir='./DT4D',
    train_split=data_split['train_set'],
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    sample_info=cfg.sample_info,
    cache_data=cfg.cache_data,
    only_shape=cfg.only_shape,
)

if not cfg.continue_from:
    exp_name = utils.build_exp_name(
        cfg=cfg,
        extra_name=args.extra_name,
    )
else:
    print("#"*20)
    print("Continuing from...",)
    print("#"*20)
    exp_name = cfg.continue_from

# Initialize trainer
if cfg.shape_ft:
    trainer = training_ft.Trainer(
        args.debug,
        torch.device("cuda"),
        train_dataset,
        exp_dir, exp_name,
        # train_to_augmented
    )
elif cfg.pose_ft:
    trainer = training_ft_pose.Trainer(
        args.debug,
        torch.device("cuda"),
        train_dataset,
        exp_dir, exp_name,
        # train_to_augmented
    )
else:
    trainer = training_t.Trainer(
        args.debug,
        torch.device("cuda"),
        train_dataset,
        exp_dir, exp_name,
        # train_to_augmented
    )

########################################################################################################################
###################################################### PRINT PARAMS ####################################################
########################################################################################################################

print()
print("#"*60)
print("#"*60)
print()
print("exp_name:")
print(exp_name)
print()
if cfg.only_shape:
    print("LEARNING SHAPE SPACE")
else:    
    print("LEARNING POSE SPACE")
print()

print()
print("SHAPE ------------------------------")
print()
for k, v in cfg.shape_network_specs.items(): print(f"{k:<40}, {v}")
print()
print("POSE  ------------------------------")
print()
for k, v in cfg.pose_network_specs.items(): print(f"{k:<40}, {v}")

print()
print("#"*60)
print("#"*60)

# if not args.debug:
#     print()
#     input("Verify the params above and press enter if you want to go ahead...")

########################################################################################################################
########################################################################################################################

trainer.train_model()

print()
print("Training done!")
print()