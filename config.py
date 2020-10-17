# path to dataset
BASE_DATASET_PATH = "./dataset/dataset/"
CONCATENATE_PATH = BASE_DATASET_PATH + "concatenate-all/"

# path where we put the partitioned datased
TRAIN_OUTPUT_PATH = BASE_DATASET_PATH + "train/"
TEST_OUTPUT_PATH = BASE_DATASET_PATH + "test/"
VALIDATION_OUTPUT_PATH = BASE_DATASET_PATH + "validation/"

# define the amount of data that will be used training
TRAIN_SPLIT = 0.80

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.02

# PARAMETERS
BATCH_SIZE = 32
NUM_EPOCHS = 90
LR = 1e-4
MOMENTUM = 0.5

SAVE_AFTER_N_EPOCHS = 50

IMG_WIDTH = 256
IMG_HEIGHT = 256

# Checkpoints on train.py
LOAD_CHECKPOINTS = False






