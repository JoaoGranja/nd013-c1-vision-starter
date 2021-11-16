import argparse
import glob
import os
import random
import re
import math

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    
    data_dir  = data_dir.replace('\\', '/')
    source    = os.path.join(data_dir, 'training_and_validation')
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')
    test_dir  = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.tfrecord)$', f)]

    num_images = len(images)
    num_val_images   = math.ceil(0.2*num_images)
    num_test_images  = math.ceil(0.1*num_images)

    for i in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        os.rename(os.path.join(source, filename),
                 os.path.join(test_dir, filename))
        images.remove(images[idx])

    for i in range(num_val_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        os.rename(os.path.join(source, filename),
                 os.path.join(val_dir, filename))
        images.remove(images[idx])
        
    for filename in images:
        os.rename(os.path.join(source, filename),
                 os.path.join(train_dir, filename))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)