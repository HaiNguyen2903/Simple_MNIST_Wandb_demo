import wandb
from model import Net
from utils import get_dataloader, get_dataset, get_transform

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from config import *
import os

def test(model, epoch, testloader):
    model.eval()
    test_loss = 0
    correct   = 0
    best_acc = 0

    my_table = wandb.Table(columns=["id", "image", "labels", "prediction"])

    with torch.no_grad():
        for idx, (data, target) in enumerate(testloader):
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            predict = output.data.max(1, keepdim=True)[1]
            correct += predict.eq(target.view_as(predict)).sum().item()

            my_table.add_data(idx, wandb.Image(data), target, predict)
    
    test_loss /= len(testloader)
    test_accuracy = 100. * correct / len(testloader.dataset)

    wandb.log({'test/loss':test_loss, 'test/accuracy': test_accuracy})

    # Save checkpoint with pytorch
    if not os.path.exists(CKPT_PATH):
        os.mkdir(CKPT_PATH)

    if test_accuracy > best_acc:
        torch.save(model.state_dict(), CKPT_PATH + 'best.pth')
        # wandb.save(os.path.join(CKPT_PATH, 'best.pth'))

    torch.save(model.state_dict(), CKPT_PATH + 'last.pth')
    # wandb.save(os.path.join(CKPT_PATH, 'last.pth'))

    '''
    Log checkpoint artifact with wandb
    '''
    ckpt_best_artifact = wandb.Artifact('best.pth', type = 'checkpoint')
    ckpt_best_artifact.add_file(os.path.join(CKPT_PATH, 'best.pth'))

    ckpt_last_artifact = wandb.Artifact('last.pth', type = 'checkpoint')
    ckpt_last_artifact.add_file(os.path.join(CKPT_PATH, 'last.pth'))

    wandb.run.log_artifact(ckpt_last_artifact, aliases=['last'])
    wandb.run.log_artifact(ckpt_best_artifact, aliases=['best'])
    # wandb.log({"predict table": my_table})

    return test_loss, test_accuracy