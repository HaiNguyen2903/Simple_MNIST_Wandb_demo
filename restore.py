import torch
import wandb
from config import *
from IPython import embed
config = dict(
        learning_rate = LEARNING_RATE,
        momentum      = MOMENTUM,
        architecture  = ARCHITECTURE,
        dataset       = DATASET
    )

# run = wandb.init(project = 'demo_wandb', job_type = "dataset-creation", tags=['create_dataset, add config'], config=config)
# artifact = wandb.Artifact('my-dataset', type='dataset')
# artifact.add_file('data/test.txt')
# run.log_artifact(artifact, aliases=['new_data', 'latest'])

# ckpt_artifact = wandb.Artifact('best.pth', type = 'checkpoint')
# ckpt_artifact.add_file('checkpoints/best.pth')
# run.log_artifact(ckpt_artifact)

import wandb
run = wandb.init()
artifact = run.use_artifact('hainguyen/demo_wandb/best.pth:v0', type='checkpoint')
artifact_dir = artifact.download('new/')

embed()

