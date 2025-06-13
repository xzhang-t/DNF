#################################################
# SET ME
#################################################
# config_dataset = "DT4D"
config_dataset = "Latent"
#################################################

splits_dir = "ZSPLITS"

if config_dataset == "DT4D":
    from configs_train.config_train_DT4D import *
elif config_dataset == "Latent":
    from configs_train.config_train_Latent import *
else:
    raise Exception('bad config dataset')

