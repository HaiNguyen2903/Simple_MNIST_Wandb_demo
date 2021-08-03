from config import PROJECT_NAME
import os
import wandb
from config import *

config = dict(
        learning_rate = LEARNING_RATE,
        momentum      = MOMENTUM,
        architecture  = ARCHITECTURE,
        dataset       = DATASET
    )

class WandbLogger():
    def __init__(self, project_name = PROJECT_NAME, config = config, tags = 'training'):
        # self.wandb, self.run = wandb if not wandb else wandb.run
        self.project_name = project_name
        self.config = config

        self.run = wandb.init(project=self.project_name, config = self.config, tags = tags) 


    
    def log_ckpt_artifact(self, ckpt_path, alias):
        assert alias in ['best', 'last']

        if alias == 'best':
            ckpt_artifact = wandb.Artifact('best.pth', type = 'checkpoint')
            ckpt_artifact.add_file(ckpt_path)
            self.run.log_artifact(ckpt_artifact, aliases=alias)

        if alias == 'last':
            ckpt_artifact = wandb.Artifact('last.pth', type = 'checkpoint')
            ckpt_artifact.add_file(ckpt_path)
            self.run.log_artifact(ckpt_artifact, aliases=alias)


    def log_dataset_artifact(self, data_dir, data_name=DATASET):
        dataset_artifact = wandb.Artifact(data_name, type = 'dataset')
        dataset_artifact.add_dir(data_dir)
        self.run.log_artifact(dataset_artifact)
        

