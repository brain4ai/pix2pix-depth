# import the necessary packages
import config
from imutils import paths
import random
import shutil
import os

# grab the paths to all input images
concatenate_Paths = list(paths.list_images(config.CONCATENATE_PATH))
random.seed(42)
random.shuffle(concatenate_Paths)

# compute the training and testing split
i = int(len(concatenate_Paths) * config.TRAIN_SPLIT)
images_trainPaths = concatenate_Paths[:i]
images_testPaths = concatenate_Paths[i:]

# we'll be using part of the training data for validation
i = int(len(images_trainPaths) * config.VAL_SPLIT)
images_valPaths = images_trainPaths[:i]
images_trainPaths = images_trainPaths[i:]

print(images_valPaths)

print("[INFO] Total images on dataset : " + str(len(concatenate_Paths)))
print("[INFO] Images for train : " + str(len(images_trainPaths)))
print("[INFO] Images for test : " + str(len(images_testPaths)))
print("[INFO] Images for validation : " + str(len(images_valPaths)))

for p in images_trainPaths:
    # obtain file name
    filename = p.split("/")[-1]

    # create the directory for train images if not exist
    if not os.path.exists(config.TRAIN_OUTPUT_PATH):
        print("[INFO] 'creating {}' directory".format(config.TRAIN_OUTPUT_PATH))
        os.makedirs(config.TRAIN_OUTPUT_PATH)

    # construct the path to the destination image and then copy
    # the image itself
    im_path = config.TRAIN_OUTPUT_PATH + filename
    shutil.copy2(p, im_path)

for p in images_testPaths:
    # obtain file name
    filename = p.split("/")[-1]

    # create the directory for train images if not exist
    if not os.path.exists(config.TEST_OUTPUT_PATH):
        print("[INFO] 'creating {}' directory".format(config.TEST_OUTPUT_PATH))
        os.makedirs(config.TEST_OUTPUT_PATH)

    # construct the path to the destination image and then copy
    # the image itself
    im_path = config.TEST_OUTPUT_PATH + filename
    shutil.copy2(p, im_path)

for p in images_valPaths:
    # obtain file name
    filename = p.split("/")[-1]

    # create the directory for train images if not exist
    if not os.path.exists(config.VALIDATION_OUTPUT_PATH):
        print("[INFO] 'creating {}' directory".format(config.VALIDATION_OUTPUT_PATH))
        os.makedirs(config.VALIDATION_OUTPUT_PATH)

    # construct the path to the destination image and then copy
    # the image itself
    im_path = config.VALIDATION_OUTPUT_PATH + filename
    shutil.copy2(p, im_path)
