import os
import argparse
import torch
import datasets.sdf_dataset_4d_new as sdf_dataset
from datasets.latent_dataset import LatentDataset, FTDataset
import json
import utils.utils as utils
import utils.nnutils as nnutils
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data_utils
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

import configs_train.config_train_Latent as cfg
from transformer import Transformer
from hyperdiffusion import HyperDiffusion

cfg.config_dataset = "Latent"
print(f"Loaded {cfg.config_dataset} config. Continue?")


parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-n', '--extra_name', default="", type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

exp_dir = os.path.join(cfg.exp_dir, cfg.exp_version)
device = torch.device("cuda")

print()
print("EXP_DIR:", exp_dir)
print()

if not cfg.continue_from:
    exp_name = utils.build_exp_name(
        cfg=cfg,
        extra_name=args.extra_name,
    )
else:
    print("#" * 20)
    print("Continuing from...", )
    print("#" * 20)
    exp_name = cfg.continue_from

print("EXP_name",exp_name)

exp_path = os.path.join(exp_dir, exp_name)

logger = TensorBoardLogger("tb_logs", name=exp_name)

print("TRAIN DATASET...")


splits_dir = f'./animals/{cfg.train_dataset_name}.json'
print("splits dir", splits_dir)
with open(splits_dir, "r") as f:
    data_split = json.load(f)

if cfg.with_ft: # using both latent and sigma as representation
    sigma_length = cfg.sigma_length
    if cfg.only_shape:
        train_dataset = FTDataset(
            cfg.shape_dir,
            cfg.checkpoint,
            cfg.ft_shape_dir,
            data_split['train_set'],
            sigma_length,
            only_shape=True,
        )
    else:
        train_dataset = FTDataset(
            cfg.shape_dir,
            cfg.checkpoint,
            cfg.ft_pose_dir,
            data_split['train_set'],
            sigma_length,
            cfg.only_shape,
            cfg.pose_dir,
            cfg.add_dir,
            cfg.n_frame,
            cfg.diff_frame,
            cfg.reverse,
        )
else: # using only global latent as representation
    if cfg.only_shape:
        train_dataset = LatentDataset(
            cfg.shape_dir,
            cfg.checkpoint,
            data_split['train_set'],
            only_shape=True,
        )
    else:
        train_dataset = LatentDataset(
            cfg.shape_dir,
            cfg.checkpoint,
            data_split['train_set'],
            cfg.only_shape,
            cfg.pose_dir,
            cfg.n_frame,
            cfg.diff_frame,
        )

print("Train dataset length: ", len(train_dataset))

test_dataset = LatentDataset(
            cfg.shape_dir,
            cfg.checkpoint,
            data_split['test_set'],
            only_shape=True,
        )

train_dl = data_utils.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_data_loader_threads,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True
    )

val_dl = data_utils.DataLoader(
        torch.utils.data.Subset(train_dataset, [0]), batch_size=1, shuffle=False, num_workers=0,
    )
test_dl = data_utils.DataLoader(
        test_dataset, batch_size=12, shuffle=True, num_workers=0
    )

layers = []
layer_names = []


if cfg.only_shape:
    layers.append(cfg.shape_codes_dim)
    layer_names.append('shape_latent')
else:
    layers.append(cfg.pose_codes_dim)
    layer_names.append('pose_latent')

if cfg.with_ft:
    for i, dim in enumerate(sigma_length):
        layers.append(dim)
        layer_names.append('sigma_' + str(i))

print(layers)
print(layer_names)


input_data = next(iter(train_dl))
# ### with ft [32, 1, 384]; [32, 4, 10576]; [32]
# ### no ft [32, 1, 384]; [32, 4, 384]; [32]
# print("data")
# for item in input_data:
#     print(item.shape)

model = Transformer(
            layers, layer_names, **cfg.transformer_config).to(device)

if cfg.only_shape:
    diffuser = HyperDiffusion(
        model, layers, input_data[0].shape, cfg
    )
else:
    diffuser = HyperDiffusion(
        model, layers, input_data[1].shape, cfg
    )

# Specify where to save checkpoints
checkpoint_path = os.path.join(exp_path,'checkpoints')
os.makedirs(checkpoint_path,exist_ok=True)
model_resume_path = cfg.model_resume_path

last_model_saver = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="last-{epoch:02d}-{train_loss:.2f}",
        save_on_train_epoch_end=True,
    )

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=cfg.epochs,
        strategy="ddp",
        logger=logger,
        default_root_dir=checkpoint_path,
        callbacks=[
            last_model_saver,
            lr_monitor,
        ],
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.accumulate_grad_batches
    )

if cfg.mode == "train":
    model_resume_path = cfg.model_resume_path
    print("training!!!!", model_resume_path)
    trainer.fit(diffuser, train_dl, val_dl,ckpt_path=model_resume_path)

else:
    best_model_save_path = cfg.best_model_save_path
    trainer.test(
        diffuser,
        test_dl,
        ckpt_path=best_model_save_path if cfg.mode == "test" else None,
    )

logger.finalize("Success")
