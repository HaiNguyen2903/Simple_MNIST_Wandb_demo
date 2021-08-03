# hyperparameters 
BATCH_SIZE_TRAIN= 256
BATCH_SIZE_TEST = 128
MOMENTUM        = 0.9
LEARNING_RATE   = 0.001
EPOCHS          = 3
DROPOUT_RATE    = 0.1

# task config
PROJECT_NAME    = 'demo_wandb'
RUN_NAME        = 'demo_1'
ARCHITECTURE    = 'Simple CNN'
DATASET         = 'MNIST'
NUM_CLASSES     = 10
LOG_INTERVAL    = 10
CLASSES         = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

# path
DATA_PATH       = 'data/'
SAVE_PATH       = 'models/' + RUN_NAME
CKPT_PATH       = 'checkpoints/'

