import wandb
from model import Net
from utils import get_dataloader, get_dataset, get_transform

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from config import *

from test import *

from IPython import embed

def train(model, epoch, trainloader, optimizer, loss_function):

    wandb.watch(model, log_freq=10)
    
    model.train()
    running_loss = 0

    for i, (input, target) in enumerate(trainloader):
        # zero the gradient
        optimizer.zero_grad()

        # forward + backpropagation + step
        predict = model(input)
        loss = loss_function(predict, target)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()

    total_loss = running_loss/len(trainloader.dataset)

    wandb.log({'train loss':total_loss})
    wandb.log({'input': wandb.Image(input)})
    

    # wandb save as artifact
    torch.onnx.export(model, input, RUN_NAME+'.onnx')
    wandb.save(RUN_NAME+'.onnx')
    trained_weight = wandb.Artifact("CNN", type="model", description="test")
    trained_weight.add_file(RUN_NAME+'.onnx')
    run.log_artifact(trained_weight)

    # pytorch save
    # torch.save(model.state_dict(), SAVE_PATH+'.pth')
    embed()
    return 


if __name__ == '__main__':
    # init wandb
    config = dict(
        learning_rate = LEARNING_RATE,
        momentum      = MOMENTUM,
        architecture  = ARCHITECTURE,
        dataset       = DATASET
    )

    run = wandb.init(project="demo_wandb", tags=["cnn"], config=config)
    
    artifact = wandb.Artifact('mnist', type='dataset')

    artifact.add_dir(DATA_PATH)

    run.log_artifact(artifact)

    # get dataloader
    train_set, test_set = get_dataset(transform=get_transform())
    trainloader, testloader = get_dataloader(train_set=train_set, test_set=test_set)

    # create model
    model = Net()

    # define optimizer and loss function
    epochs = EPOCHS
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer     = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # training
    pb = tqdm(range(epochs))
    train_losses, test_losses, test_accuracy = [], [], []

    for epoch in pb:
        train_loss = train(model, epoch, trainloader, optimizer, loss_function)
        train_losses.append(train_loss)
        print(train_loss)
        test_loss, test_acc = test(model, epoch, testloader)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)